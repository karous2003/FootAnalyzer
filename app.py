import cv2
import numpy as np
from rembg import remove

def scale_point(point, scale):
    return (int(point[0] / scale), int(point[1] / scale))

def process_insole(input_image):
    scale_factor = 0.5  # 縮小比例（0.5 表示縮小 50%）

    # 先縮小圖片
    small_image = cv2.resize(input_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # 去背
    small_output_image = remove(small_image)
    if small_output_image.shape[2] == 4:
        small_output_image = cv2.cvtColor(small_output_image, cv2.COLOR_BGRA2BGR)

    # 轉灰階、強化對比
    gray = cv2.cvtColor(small_output_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=6, beta=0)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 150, 220)

    # 找輪廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "Error: No contour detected", None

    # 獲取最大輪廓
    max_contour = max(contours, key=cv2.contourArea)

    # 放大回原圖大小
    output_image = cv2.resize(small_output_image, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 恢復點的座標（縮放後需還原）
    bottom_point = scale_point(tuple(max_contour[max_contour[:, :, 1].argmax()][0]), scale_factor)
    front_point = scale_point(tuple(max_contour[max_contour[:, :, 1].argmin()][0]), scale_factor)

    # 繪製結果
    cv2.circle(output_image, bottom_point, 10, (0, 0, 255), -1)
    cv2.circle(output_image, front_point, 10, (255, 0, 0), -1)
    cv2.line(output_image, bottom_point, front_point, (0, 255, 0), 3)

    return output_image
