from ultralytics import YOLO
import cv2

selected_route = []
laser_px = None
laser_chosen = False
laser_temp = None

def mouse_callback(event, x, y, flags, param):
    global laser_temp, laser_chosen
    if event == cv2.EVENT_LBUTTONDOWN:
        if not laser_chosen:
            laser_temp = (x, y)
            print(f"⚪ 임시 레이저 후보점: {laser_temp}")
