# A_click_select.py
import cv2
import time
import numpy as np

CYCLE_WINDOW_SEC = 0.5   # 같은 자리 연속 클릭으로 후보 순환 인정 시간
PIXEL_TOL = 4            # 같은 자리 판단 허용 오차(px)

class BiClickSelector:
    def __init__(self, holdsL, holdsR, W, swap_display=False):
        self.holdsL = holdsL
        self.holdsR = holdsR
        self.W = W
        self.swap = swap_display
        # 선택 인덱스(선택 순서 유지)
        self.selL = []
        self.selR = []

        # 겹침 처리용 면적/컨투어 캐시
        self._cache_hold_geom(self.holdsL)
        self._cache_hold_geom(self.holdsR)

        # 같은 자리 연속 클릭 시 후보 순환 상태 (좌/우 각각 유지)
        self._lastL = {"pt": None, "cands": [], "i": 0, "t": 0.0}
        self._lastR = {"pt": None, "cands": [], "i": 0, "t": 0.0}

    @staticmethod
    def _cache_hold_geom(holds):
        for h in holds:
            cnt = np.asarray(h["contour"], dtype=np.int32)
            if cnt.ndim == 2:  # (N,2) -> (N,1,2)
                cnt = cnt.reshape(-1, 1, 2)
            h["_cnt"] = cnt
            h["_area"] = float(cv2.contourArea(cnt))

    def _hit_test_cycle(self, holds, x, y, last_state):
        """
        클릭 좌표 (x,y)에 포함되는 모든 후보를 면적 오름차순으로 정렬.
        같은 자리(시간/위치)면 다음 후보로 순환해 단일 인덱스를 반환.
        """
        pt = (int(x), int(y))
        # 1) 후보 모두 수집 (안쪽 포함)
        cands = []
        for i, h in enumerate(holds):
            if cv2.pointPolygonTest(h["_cnt"], pt, False) >= 0:
                cands.append((i, h["_area"]))
        if not cands:
            # 빈 공간 클릭 → 상태 리셋
            last_state.update({"pt": None, "cands": [], "i": 0, "t": 0.0})
            return None

        # 2) “안쪽 우선” 면적 작은 순으로 정렬
        cands.sort(key=lambda z: z[1])
        cand_indices = [i for i, _ in cands]

        # 3) 같은 자리 빠른 재클릭이면 후보 순환
        now = time.time()
        same_spot = False
        if last_state["pt"] is not None:
            dx = abs(last_state["pt"][0] - pt[0])
            dy = abs(last_state["pt"][1] - pt[1])
            same_spot = (dx <= PIXEL_TOL and dy <= PIXEL_TOL and (now - last_state["t"]) <= CYCLE_WINDOW_SEC)

        if same_spot and last_state["cands"]:
            # 직전 후보 목록과 이번 목록이 동일한지 확인(겹침 세트가 같을 때만 순환)
            if cand_indices == last_state["cands"]:
                last_state["i"] = (last_state["i"] + 1) % len(cand_indices)
            else:
                last_state["i"] = 0
        else:
            last_state["pt"] = pt
            last_state["i"] = 0

        last_state["cands"] = cand_indices
        last_state["t"] = now
        return cand_indices[last_state["i"]]

    def _side_and_local_xy(self, x, y):
        # 합성(vis) 기준 x를 좌/우 및 로컬 좌표로 변환
        if not self.swap:
            # [0..W-1]: L, [W..2W-1]: R
            if 0 <= x < self.W:
                return "L", x, y
            elif self.W <= x < 2*self.W:
                return "R", x - self.W, y
        else:
            # 스왑: [0..W-1]: R, [W..2W-1]: L
            if 0 <= x < self.W:
                return "R", x, y
            elif self.W <= x < 2*self.W:
                return "L", x - self.W, y
        return None, None, None

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        side, lx, ly = self._side_and_local_xy(x, y)
        if side is None:
            return

        if side == "L":
            idx = self._hit_test_cycle(self.holdsL, lx, ly, self._lastL)
            if idx is not None:
                if idx in self.selL:
                    self.selL.remove(idx)
                else:
                    self.selL.append(idx)
        else:
            idx = self._hit_test_cycle(self.holdsR, lx, ly, self._lastR)
            if idx is not None:
                if idx in self.selR:
                    self.selR.remove(idx)
                else:
                    self.selR.append(idx)


def interactive_select_live_left_only(cap, holds, window="D455"):

    # ---- 준비: 컨투어/면적 캐시 ----
    for h in holds:
        cnt = np.asarray(h["contour"], dtype=np.int32)
        if cnt.ndim == 2: cnt = cnt.reshape(-1,1,2)
        h["_cnt"]  = cnt
        h["_area"] = float(cv2.contourArea(cnt))

    selL = []
    lastL = {"pt": None, "cands": [], "i": 0, "t": 0.0}
    H = W = None

    def _hit_test_cycle(x, y):
        pt = (int(x), int(y))
        cands = []
        for i, h in enumerate(holds):
            if cv2.pointPolygonTest(h["_cnt"], pt, False) >= 0:
                cands.append((i, h["_area"]))
        if not cands:
            lastL.update({"pt": None, "cands": [], "i": 0, "t": 0.0})
            return None
        cands.sort(key=lambda z: z[1])
        cand_indices = [i for i,_ in cands]
        now = time.time()
        same_spot = False
        if lastL["pt"] is not None:
            dx = abs(lastL["pt"][0] - pt[0]); dy = abs(lastL["pt"][1] - pt[1])
            same_spot = (dx <= PIXEL_TOL and dy <= PIXEL_TOL and (now - lastL["t"]) <= CYCLE_WINDOW_SEC)
        if same_spot and lastL["cands"]:
            if cand_indices == lastL["cands"]:
                lastL["i"] = (lastL["i"] + 1) % len(cand_indices)
            else:
                lastL["i"] = 0
        else:
            lastL["pt"] = pt
            lastL["i"]  = 0
        lastL["cands"] = cand_indices
        lastL["t"] = now
        return cand_indices[lastL["i"]]

    # 마우스 콜백 (단일 화면)
    def _on_mouse(event, x, y, flags, param):
        nonlocal selL
        if event != cv2.EVENT_LBUTTONDOWN: return
        if W is None or H is None: return
        idx = _hit_test_cycle(x, y)
        if idx is None: return
        if idx in selL: selL.remove(idx)
        else: selL.append(idx)

    # 같은 창 재사용
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok: break
        if W is None:
            H, W = frame.shape[:2]
        vis = frame.copy()

        # 모든 컨투어 얇게(검정)
        for h in holds:
            cv2.drawContours(vis, [h["_cnt"]], -1, (0,0,0), 2, cv2.LINE_AA)

        # 선택 컨투어 강조 + L1/L2…
        for n, i in enumerate(selL, start=1):
            h = holds[i]
            cv2.drawContours(vis, [h["_cnt"]], -1, (0,255,255), 3, cv2.LINE_AA)
            cx, cy = h["center"]
            cv2.putText(vis, f"L{n}", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"L{n}", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 1, cv2.LINE_AA)

        msg = "[Click] toggle (겹치면 순환)  |  [Space] finish  |  [R] reset  |  [Q/ESC] cancel"
        cv2.putText(vis, msg, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, msg, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 1, cv2.LINE_AA)

        cv2.imshow(window, vis)
        k = cv2.waitKey(10) & 0xFF
        if k == ord(' '):       # finish
            break
        elif k in (ord('q'), 27):
            selL = []
            break
        elif k in (ord('r'), ord('R')):
            selL = []

    # 창은 유지(이후 메인 루프로 바로 전환). 필요하면 여기서 clear만.
    return selL
