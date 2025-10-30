# mediapipe_pose_utils.py
import time
import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    _HAS_MP = False

class PoseTracker:
    def __init__(self, min_detection_confidence=0.5, model_complexity=1):
        self.enabled = _HAS_MP
        if not self.enabled:
            self.pose = None
            return
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            model_complexity=model_complexity
        )
        # 기본 사용 랜드마크
        self.important_landmarks = {
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "mouth_left": 9,
        "mouth_right": 10,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_pinky": 17,
        "right_pinky": 18,
        "left_index": 19,
        "right_index": 20,
        "left_thumb": 21,
        "right_thumb": 22,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
        }

        self.hand_parts = {
            "left_wrist","right_wrist",
            "left_index","right_index",
            "left_thumb","right_thumb",
            "left_pinky","right_pinky"}

        # 발로 인정되는 파트(ankle 제외)
        self.foot_grip_parts = {"left_heel","right_heel","left_foot_index","right_foot_index"}

        # 최종 grip 파트
        self.grip_parts = self.hand_parts | self.foot_grip_parts

        # 차폐 파트(= 중요 랜드마크 전체 - grip)
        self.blocking_parts = set(self.important_landmarks.keys()) - self.grip_parts

    def process(self, bgr_image):
        """
        bgr_image: (H,W,3)
        return: dict{name->(x_px, y_px)}; mediapipe 미사용/실패 시 {}
        """
        if not self.enabled or self.pose is None or bgr_image is None:
            return {}
        h, w = bgr_image.shape[:2]
        res = self.pose.process(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        if not res or not res.pose_landmarks:
            return {}
        coords = {}
        min_vis = 0.5
        for name, idx in self.important_landmarks.items():
            lm = res.pose_landmarks.landmark[idx]
            if getattr(lm, "visibility", 1.0) < min_vis:
                continue
            x = max(0, min(w-1, lm.x * w))
            y = max(0, min(h-1, lm.y * h))
            coords[name] = (x, y)
        return coords

    def close(self):
        if self.pose is not None:
            self.pose.close()
            self.pose = None

def draw_pose_points(vis, coords, offset_x=0):
    allow = {
        # 손
        "left_wrist", "right_wrist",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_pinky", "right_pinky",
        # 발
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    }

    for name, (x, y) in coords.items():
        if name not in allow:
            continue  # 손·발 이외는 스킵

        # 손(빨강) / 발(초록)
        joint_color = (0, 0, 255) if name in {
            "left_wrist", "right_wrist",
            "left_index", "right_index",
            "left_thumb", "right_thumb",
            "left_pinky", "right_pinky"
        } else (0, 255, 0)

        cv2.circle(vis, (int(x) + offset_x, int(y)), 6, joint_color, -1)

def _draw_capsule(mask, p1, p2, radius_px):
    """선분(p1-p2)을 중심으로 한 캡슐(둥근 직사각형) 채우기"""
    p1 = tuple(map(int, map(round, p1)))
    p2 = tuple(map(int, map(round, p2)))
    r  = int(round(float(radius_px)))
    if r <= 0:
        return
    cv2.line(mask, p1, p2, 255, thickness=2*r, lineType=cv2.LINE_AA)
    cv2.circle(mask, p1, r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, p2, r, 255, -1, cv2.LINE_AA)


def _convex_hull_mask(points_xy, shape_hw, dilate_px=12):
    """points_xy: [(x,y), ...], shape_hw=(H,W)
       얼굴 점들의 convex hull을 마스크로 만들고 살짝 팽창해서 여유를 줌."""
    H, W = shape_hw
    if len(points_xy) < 3:
        return None
    pts = np.array(points_xy, dtype=np.int32).reshape(-1,1,2)
    hull = cv2.convexHull(pts)
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    if dilate_px and dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        m = cv2.dilate(m, k, iterations=1)
    return m

def head_mask_from_coords(coords, shape_hw):
    """coords(dict)에서 head_parts에 해당하는 점만 모아서 head 마스크 생성"""
    head_keys = {
        "nose",
        "left_eye_inner","left_eye","left_eye_outer",
        "right_eye_inner","right_eye","right_eye_outer",
        "left_ear","right_ear","mouth_left","mouth_right",
    }
    pts = [coords[k] for k in head_keys if k in coords]
    return _convex_hull_mask(pts, shape_hw, dilate_px=12)

def build_body_masks(coords, shape_hw):
    """
    coords: PoseTracker.process가 반환한 dict{name:(x,y)}
    shape_hw: (H, W)
    return: torso_mask(uint8), limb_masks(dict[name->uint8]), handsfeet_mask(uint8), head_mask(uint8)
    """
    H, W = shape_hw
    def has(*names): return all(n in coords for n in names)

    torso = np.zeros((H, W), np.uint8)
    limbs = {}  # name -> mask

    # --- Torso: 확장 사변형 + 척추 캡슐 ---
    if has("left_shoulder","right_shoulder","left_hip","right_hip"):
        Ls = np.array(coords["left_shoulder"], dtype=float)
        Rs = np.array(coords["right_shoulder"], dtype=float)
        Lh = np.array(coords["left_hip"], dtype=float)
        Rh = np.array(coords["right_hip"], dtype=float)

        Ws = np.linalg.norm(Rs - Ls) + 1e-6
        Wh = np.linalg.norm(Rh - Lh) + 1e-6

        def expand_edge(a, b, margin):
            v = b - a
            n = np.array([-v[1], v[0]], dtype=float)
            n /= (np.linalg.norm(n) + 1e-6)
            return (a + n*margin, b + n*margin)

        m_top = 0.15 * Ws
        m_bot = 0.15 * Wh
        Ls_ex, Rs_ex = expand_edge(Ls, Rs,  m_top)
        Lh_ex, Rh_ex = expand_edge(Lh, Rh, -m_bot)  # 아래쪽은 반대방향

        poly = np.array([Ls_ex, Rs_ex, Rh_ex, Lh_ex], dtype=np.int32).reshape(-1,1,2)
        cv2.fillPoly(torso, [poly], 255, cv2.LINE_AA)

        spine_top = 0.5*(Ls + Rs)
        spine_bot = 0.5*(Lh + Rh)
        r_spine = 0.25 * min(Ws, Wh)
        _draw_capsule(torso, spine_top, spine_bot, r_spine)

    # --- Limb capsules ---
    def limb_capsule(name, a, b, scale):
        if not has(a,b): return
        A = np.array(coords[a], dtype=float)
        B = np.array(coords[b], dtype=float)
        L = np.linalg.norm(B - A)
        r = float(scale) * float(L)
        m = np.zeros((H, W), np.uint8)
        _draw_capsule(m, A, B, r)
        limbs[name] = m

    limb_capsule("left_upper_arm",  "left_shoulder", "left_elbow", 0.18)
    limb_capsule("right_upper_arm", "right_shoulder","right_elbow",0.18)
    limb_capsule("left_forearm",    "left_elbow",    "left_wrist", 0.15)
    limb_capsule("right_forearm",   "right_elbow",   "right_wrist",0.15)
    limb_capsule("left_thigh",      "left_hip",      "left_knee",  0.22)
    limb_capsule("right_thigh",     "right_hip",     "right_knee", 0.22)
    limb_capsule("left_shin",       "left_knee",     "left_ankle", 0.18)
    limb_capsule("right_shin",      "right_knee",    "right_ankle",0.18)

    # --- 손/발 마스크 구성: 손=원, 발=캡슐(heel↔foot_index) ---
    hands_only = np.zeros((H, W), np.uint8)
    for n in ("left_wrist","right_wrist",
              "left_index","right_index","left_thumb","right_thumb","left_pinky","right_pinky"):
        if n in coords:
            x,y = map(int, map(round, coords[n]))
            cv2.circle(hands_only, (x,y), 16, 255, -1, cv2.LINE_AA)

    foot_mask = np.zeros((H, W), np.uint8)

    def draw_foot(heel_name, index_name):
        if has(heel_name, index_name):
            A = np.array(coords[heel_name], dtype=float)
            B = np.array(coords[index_name], dtype=float)
            L = np.linalg.norm(B - A)
            # 길이 비례 반경(필요시 상수로도 OK: ex. r=16)
            r = max(12.0, 0.22 * L)
            _draw_capsule(foot_mask, A, B, r)

    draw_foot("left_heel",  "left_foot_index")
    draw_foot("right_heel", "right_foot_index")

    # 손+발 통합
    hands_feet = cv2.bitwise_or(hands_only, foot_mask)

    head = head_mask_from_coords(coords, shape_hw)
    if head is None:
        head = np.zeros((H, W), np.uint8)

    return torso, limbs, hands_feet, head

def classify_occluder(center_xy, coords, hold_mask=None, shape_hw=None):
    # shape_hw가 없으면 hold_mask에서 추론
    if shape_hw is None:
        if hold_mask is not None:
            shape_hw = hold_mask.shape[:2]
        else:
            raise ValueError("shape_hw 또는 hold_mask 중 하나는 필요합니다.")

    # 필요 마스크 4개만 쓰면 됨(발은 hands_feet에 이미 포함)
    torso_m, limb_ms, handsfeet_m, head_m = build_body_masks(coords, shape_hw)

    # 1) 홀드∩손/발 겹치면 grip 우선
    if hold_mask is not None:
        overlap_any = cv2.bitwise_and(handsfeet_m, hold_mask)
        if np.count_nonzero(overlap_any) > 0:
            return ("grip", "hand/foot")

    # 2) head / limb / torso 차단
    H, W = shape_hw
    cx, cy = map(int, map(round, center_xy))
    cx = max(0, min(W-1, cx)); cy = max(0, min(H-1, cy))
    if head_m[cy, cx] > 0:
        return ("blocked", "head")
    for lname, m in limb_ms.items():
        if m[cy, cx] > 0:
            return ("blocked", lname)
    if torso_m[cy, cx] > 0:
        return ("blocked", "torso")

    # 3) 폴백: grip 파트 제외한 가장 가까운 관절
    GRIP_PARTS = {
        # 손
        "left_wrist","right_wrist","left_index","right_index",
        "left_thumb","right_thumb","left_pinky","right_pinky",
        # 발
        "left_heel","right_heel","left_foot_index","right_foot_index",
    }
    cand = [n for n in coords.keys() if n not in GRIP_PARTS]
    best, best_d2 = None, 1e18
    for n in cand:
        x,y = coords[n]
        d2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
        if d2 < best_d2:
            best_d2, best = d2, n

    HEAD_KEYS = {
        "nose","left_eye_inner","left_eye","left_eye_outer",
        "right_eye_inner","right_eye","right_eye_outer",
        "left_ear","right_ear","mouth_left","mouth_right",
    }
    if best in HEAD_KEYS:
        return ("blocked", "head")
    return ("blocked", best or "unknown")
