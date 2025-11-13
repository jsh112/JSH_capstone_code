SELECTED_COLOR = None    # 예: 'orange' (None=전체)

# === (NEW) 웹 모듈 - 색상 선택 ===
_USE_WEB = True
try:
    from A_web import choose_color_via_web
except Exception:
    _USE_WEB = False
    def choose_color_via_web(*a, **k):
        raise RuntimeError("color_web 모듈(A_web)이 로드되지 않았습니다.")


# ==== 색상 맵 ====
ALL_COLORS = {
    'red':'Hold_Red','orange':'Hold_Orange','yellow':'Hold_Yellow','green':'Hold_Green',
    'blue':'Hold_Blue','purple':'Hold_Purple','pink':'Hold_Pink','white':'Hold_White',
    'black':'Hold_Black','gray':'Hold_Gray','lime':'Hold_Lime','sky':'Hold_Sky',
}

def _sanitize_label(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch in ("_", "-"))

def ask_color_and_map_to_class(all_colors_dict):
    print("가능한 색상:", ", ".join(all_colors_dict.keys()))
    s = input("필터할 색상 입력(엔터=전체): ").strip().lower()
    if not s:
        print("→ 전체 표시 사용")
        return None, "all"   # (모델클래스=None, 파일라벨="all")
    mapped = all_colors_dict.get(s)
    if mapped is None:
        print(f"입력 '{s}' 은(는) 유효하지 않은 색입니다. 전체 표시 사용")
        return None, "all"
    print(f"선택된 클래스명: {mapped}")
    return mapped, s        # (모델클래스, 파일라벨)

def choose_color(args):
    # ================= 색상 필터 선택 =================
    selected_class_name  = None     # 모델 클래스명 (예: Hold_Green)
    selected_color_label = "all"    # 파일명 라벨 (예: green, orange, all)

    # 1) 웹에서 색상 선택(가능하고 --no_web가 아닐 때)
    if (not args.no_web) and _USE_WEB:
        try:
            chosen = choose_color_via_web(
                all_colors=list(ALL_COLORS.keys()),
                defaults={"port": args.port, "baud": args.baud}
            )  # ""이면 전체
            if chosen:
                mapped = ALL_COLORS.get(chosen)
                if mapped is None:
                    print(f"[Filter] 웹 선택 '{chosen}' 무효 → 전체 표시")
                else:
                    print(f"[Filter] 웹 선택: {chosen} → {mapped}")
                    selected_class_name  = mapped
                    selected_color_label = chosen.lower()
            else:
                print("[Filter] 웹에서 전체 선택")
        except Exception as e:
            print(f"[Filter] 웹 선택 실패 → 콘솔 대체: {e}")

    # 2) 고정 설정 값
    if (selected_class_name is None) and (SELECTED_COLOR is not None):
        sc = SELECTED_COLOR.strip().lower()
        mapped = ALL_COLORS.get(sc)
        if mapped is None:
            print(f"[Filter] SELECTED_COLOR='{SELECTED_COLOR}' 무효 → 콘솔에서 선택")
            selected_class_name, selected_color_label = ask_color_and_map_to_class(ALL_COLORS)
        else:
            print(f"[Filter] 고정 선택 클래스: {mapped}")
            selected_class_name  = mapped
            selected_color_label = sc

    # 3) 콘솔 입력 대체
    if selected_class_name is None and (args.no_web or not _USE_WEB):
        selected_class_name, selected_color_label = ask_color_and_map_to_class(ALL_COLORS)

    # === 여기서 색상 라벨에 맞춰 CSV 파일명 생성 ===
    csv_label = _sanitize_label(selected_color_label) if selected_color_label else "all"
    CSV_GRIPS_PATH_dyn = f"grip_records_{csv_label}.csv"
    print(f"[Info] 경로 CSV: {CSV_GRIPS_PATH_dyn}")
    return selected_class_name, CSV_GRIPS_PATH_dyn
