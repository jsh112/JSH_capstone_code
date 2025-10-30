import cv2, numpy as np

def rotate_image(img, rot_code):
    return cv2.rotate(img, rot_code) if rot_code is not None else img

COLOR_MAP = {
    'Hold_Red':(0,0,255),'Hold_Orange':(0,165,255),'Hold_Yellow':(0,255,255),
    'Hold_Green':(0,255,0),'Hold_Blue':(255,0,0),'Hold_Purple':(204,50,153),
    'Hold_Pink':(203,192,255),'Hold_Lime':(50,255,128),'Hold_Sky':(255,255,0),
    'Hold_White':(255,255,255),'Hold_Black':(30,30,30),'Hold_Gray':(150,150,150),
}

def merge_holds_by_center(holds_lists, merge_dist_px=18):
    merged = []
    for holds in holds_lists:
        for h in holds:
            h = {k: v for k, v in h.items()}
            h.pop("hold_index", None)
            assigned = False
            for m in merged:
                dx = h["center"][0] - m["center"][0]
                dy = h["center"][1] - m["center"][1]
                if (dx*dx + dy*dy) ** 0.5 <= merge_dist_px:
                    area_h = cv2.contourArea(h["contour"])
                    area_m = cv2.contourArea(m["contour"])
                    if (area_h > area_m) or (abs(area_h - area_m) < 1e-6 and h.get("conf",0) > m.get("conf",0)):
                        m.update(h)
                    assigned = True
                    break
            if not assigned:
                merged.append(h)
    return merged

# ---------------------------
# 내부 유틸: YOLO polygon → mask 생성 (원본 좌표계)
# ---------------------------
def _build_holds_from_polys(res, frame_bgr, names, color_map=None):
    H, W = frame_bgr.shape[:2]
    out = []
    if (res.masks is None) or (not hasattr(res.masks, "xy")):
        return out

    # 각 인스턴스의 폴리곤(원본 좌표계)
    polys_all = res.masks.xy
    boxes = getattr(res, "boxes", None)

    for i, polys in enumerate(polys_all):
        if polys is None:
            continue

        # polys 가 ndarray(세그먼트 1개)일 수도, [ndarray, ...] 다중 세그먼트일 수도 있음
        segments = []
        if isinstance(polys, (list, tuple)):
            for p in polys:
                if p is None or len(p) <= 2: 
                    continue
                segments.append(np.round(p).astype(np.int32))
        else:
            if hasattr(polys, "shape") and len(polys) > 2:
                segments = [np.round(polys).astype(np.int32)]

        if not segments:
            continue

        # 1) 원본 크기의 빈 캔버스에 폴리곤 채우기
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, segments, 255)

        # 2) 컨투어/중심
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = int(np.mean(cnt[:, 0, 0]))
            cy = int(np.mean(cnt[:, 0, 1]))

        # 클래스/신뢰도
        cls_id, conf = None, 0.0
        if boxes is not None and getattr(boxes, "cls", None) is not None and i < len(boxes.cls):
            try:
                cls_id = int(boxes.cls[i].item())
            except Exception:
                cls_id = None
        if boxes is not None and getattr(boxes, "conf", None) is not None and i < len(boxes.conf):
            try:
                conf = float(boxes.conf[i].item())
            except Exception:
                conf = 0.0

        class_name = None
        if cls_id is not None:
            if isinstance(names, dict):
                class_name = names.get(cls_id, str(cls_id))
            elif isinstance(names, list) and 0 <= cls_id < len(names):
                class_name = names[cls_id]
            else:
                class_name = str(cls_id)

        color = (255, 255, 255)
        if color_map and class_name in color_map:
            color = color_map[class_name]

        out.append({
            "class_name": class_name,
            "color": color,
            "contour": cnt,
            "center": (cx, cy),
            "conf": conf,
            "mask": mask,  # ← 이후 깊이/3D에 바로 사용
        })
    return out

# ---------------------------
# 내부 유틸: raster mask(unletterbox) → 원본 크기
# ---------------------------
def _unletterbox_mask(m_proc, orig_shape, proc_shape):
    """
    m_proc: (Th, Tw) float/bool/uint8
    orig_shape: (H, W)  (원본 프레임)
    proc_shape: (Th, Tw) (모델 전처리 해상도)
    """
    H, W = orig_shape
    Th, Tw = proc_shape

    # 전처리 스케일/패딩 계산
    r = min(Tw / float(W), Th / float(H))
    new_w = int(round(W * r))
    new_h = int(round(H * r))
    pad_x = int(round((Tw - new_w) / 2.0))
    pad_y = int(round((Th - new_h) / 2.0))

    # 패딩 제거
    y0, y1 = pad_y, pad_y + new_h
    x0, x1 = pad_x, pad_x + new_w
    y0 = max(0, min(Th, y0)); y1 = max(0, min(Th, y1))
    x0 = max(0, min(Tw, x0)); x1 = max(0, min(Tw, x1))
    cropped = m_proc[y0:y1, x0:x1]

    # 원본 크기로 최근접 보간
    mask = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask

def _build_holds_from_raster(res, frame_bgr, names, mask_thresh=0.7, color_map=None):
    """res.masks.data (Th×Tw)를 사용하는 폴백 경로 — 레터박스 역변환 포함"""
    H, W = frame_bgr.shape[:2]
    out = []
    if (res.masks is None) or (not hasattr(res.masks, "data")):
        return out

    masks = res.masks.data
    boxes = getattr(res, "boxes", None)
    Th, Tw = masks.shape[1], masks.shape[2]

    for i in range(masks.shape[0]):
        m_proc = masks[i].detach().cpu().numpy()  # (Th, Tw)
        # 1) 레터박스 패딩 제거 → 원본 크기
        m = _unletterbox_mask(m_proc, (H, W), (Th, Tw))
        # 2) 이진화 + 클로징
        binary = (m > float(mask_thresh)).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = int(np.mean(cnt[:, 0, 0]))
            cy = int(np.mean(cnt[:, 0, 1]))

        cls_id, conf = None, 0.0
        if boxes is not None and getattr(boxes, "cls", None) is not None and i < len(boxes.cls):
            try:
                cls_id = int(boxes.cls[i].item())
            except Exception:
                cls_id = None
        if boxes is not None and getattr(boxes, "conf", None) is not None and i < len(boxes.conf):
            try:
                conf = float(boxes.conf[i].item())
            except Exception:
                conf = 0.0

        class_name = None
        if cls_id is not None:
            if isinstance(names, dict):
                class_name = names.get(cls_id, str(cls_id))
            elif isinstance(names, list) and 0 <= cls_id < len(names):
                class_name = names[cls_id]
            else:
                class_name = str(cls_id)

        color = (255, 255, 255)
        if color_map and class_name in color_map:
            color = color_map[class_name]

        out.append({
            "class_name": class_name,
            "color": color,
            "contour": cnt,
            "center": (cx, cy),
            "conf": conf,
            "mask": binary,  # 폴백 경로에서도 mask 포함
        })
    return out

# ===========================
# 공개 API: 초기 홀드 추출
# ===========================
def extract_holds_all_classes_no_index(frame_bgr, model, mask_thresh=0.7):
    """
    기존 기능을 보존하되, 드리프트를 없애기 위해:
      1) 우선 res.masks.xy(원본 좌표계 폴리곤)로 마스크를 생성
      2) 없으면 raster(data)를 레터박스 역변환(unletterbox)하여 원본 크기로 변환
    """
    h, w = frame_bgr.shape[:2]
    res = model(frame_bgr, imgsz=1280, verbose=False)[0]
    out = []
    if res.masks is None:
        return out

    names = getattr(model, "names", None)

    # A. 폴리곤 우선 경로 (권장)
    if hasattr(res.masks, "xy") and res.masks.xy is not None and len(res.masks.xy) > 0:
        out = _build_holds_from_polys(res, frame_bgr, names, color_map=COLOR_MAP)
        if out:
            return out

    # B. 폴리곤 미제공 시 폴백(레터박스 역변환 포함)
    out = _build_holds_from_raster(res, frame_bgr, names, mask_thresh=mask_thresh, color_map=COLOR_MAP)
    return out

# ===========================
# 초기 프레임들에서 합치기(기존 로직 그대로)
# ===========================
def initial_5frames_all_classes(cap, model, rotate_code, n_frames=5, mask_thresh=0.7, merge_dist_px=18):
    sets = []
    for _ in range(2):
        cap.read()
    for _ in range(n_frames):
        ok, f = cap.read()
        if not ok:
            continue
        f = rotate_image(f, rotate_code)
        holds = extract_holds_all_classes_no_index(f, model, mask_thresh)
        sets.append(holds)
    # 병합(중심 거리 기준, 기존 merge_holds_by_center와 유사)
    merged = []
    for holds in sets:
        for h in holds:
            assigned = False
            for m in merged:
                dx = h["center"][0] - m["center"][0]
                dy = h["center"][1] - m["center"][1]
                if (dx*dx + dy*dy) ** 0.5 <= merge_dist_px:
                    a_h = cv2.contourArea(h["contour"])
                    a_m = cv2.contourArea(m["contour"])
                    if (a_h > a_m) or (abs(a_h - a_m) < 1e-6 and h.get("conf",0) > m.get("conf",0)):
                        m.update(h)
                    assigned = True
                    break
            if not assigned:
                merged.append(dict(h))
    return merged

def assign_ids_by_selection_order(holds, selected_indices):
    selected = [holds[i] for i in selected_indices]
    for idx, h in enumerate(selected):
        h["hold_index"] = idx
    return selected

def assign_ids_by_yx(holds):
    enriched = [{"cx": h["center"][0], "cy": h["center"][1], **h} for h in holds]
    enriched.sort(key=lambda h: (h["cy"], h["cx"]))
    for i, h in enumerate(enriched):
        h["hold_index"] = i
    return enriched
