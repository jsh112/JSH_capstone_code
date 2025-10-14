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

def extract_holds_all_classes_no_index(frame_bgr, model, mask_thresh=0.7):
    h, w = frame_bgr.shape[:2]
    res = model(frame_bgr)[0]
    out = []
    if res.masks is None:
        return out
    masks = res.masks.data
    boxes = res.boxes
    names = model.names
    for i in range(masks.shape[0]):
        mask = masks[i].cpu().numpy()
        mask_rs = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (mask_rs > mask_thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        cls_id = int(boxes.cls[i].item())
        class_name = names[cls_id]
        color = COLOR_MAP.get(class_name, (255, 255, 255))
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        out.append({
            "class_name": class_name,
            "color": color,
            "contour": cnt,
            "center": (cx, cy),
            "conf": float(boxes.conf[i].item())
        })
    return out

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