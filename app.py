# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import math
import time
from flask import Flask, request, render_template, send_from_directory
from rembg import remove

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
        
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        
        output_filename, result_data = process_insole(filename)

        return render_template("index.html", image_url=output_filename, result_data=result_data)

    return render_template("index.html", image_url=None, result_data=None)

def process_insole(image_path):
    start_time = time.time()
    input_image = cv2.imread(image_path)
    
    # 去背
    output_image = remove(input_image)
    if output_image.shape[2] == 4:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGRA2BGR)

    # 轉灰階、強化對比
    gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=6, beta=0)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 150, 220)

    # 找輪廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # 獲取鞋墊的最頂點和最底點
    bottom_point = tuple(max_contour[max_contour[:, :, 1].argmax()][0])
    front_point = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
    foot_length_pixels = math.dist(bottom_point, front_point)

    # 計算 A4 紙比例換算長度（像素對實際長度）
    a4_width_cm = 21.0
    a4_length_cm = 29.7
    image_height, image_width = output_image.shape[:2]
    pixels_per_cm = ((image_width / a4_width_cm) + (image_height / a4_length_cm)) / 2

    # 計算鞋墊長度
    insole_length_cm = foot_length_pixels / pixels_per_cm

    def draw_line_and_length(image, point1, point2, length_cm, color, label, line_thickness=10):
        cv2.line(image, point1, point2, color, line_thickness)
        mid_point = (int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2))
        cv2.putText(image, f"{label}: {length_cm:.2f} cm", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 繪製足長（藍線）
    draw_line_and_length(output_image, bottom_point, front_point, insole_length_cm, (255, 135, 0), "Length")
    cv2.circle(output_image, bottom_point, 12, (255, 135, 0), -1)  # 藍色圓點
    cv2.circle(output_image, front_point, 12, (255, 135, 0), -1)  # 藍色圓點

    # 前掌寬計算（紅線）
    y_threshold = front_point[1] + int(foot_length_pixels * 0.5)
    front_half_points = [pt[0] for pt in max_contour if pt[0][1] <= y_threshold]
    if front_half_points:
        left_most = tuple(min(front_half_points, key=lambda x: x[0]))
        right_most = tuple(max(front_half_points, key=lambda x: x[0]))
        forefoot_width = math.dist(left_most, right_most) / pixels_per_cm
        draw_line_and_length(output_image, left_most, right_most, forefoot_width, (0, 0, 255), "Forefoot")
        forefoot_center = ((left_most[0] + right_most[0]) // 2, (left_most[1] + right_most[1]) // 2)
        cv2.circle(output_image, forefoot_center, 12, (0, 0, 255), -1)  # 前掌中心點

    # 中足寬計算（紫線）
    midfoot_y = int(bottom_point[1] - 0.4 * foot_length_pixels)
    midfoot_pt = (bottom_point[0], midfoot_y)

    def find_nearest_contour_point(image, start_point, direction, contour, target_y):
        x, y = start_point
        step = 5 if direction == "right" else -5
        closest_point = None
        min_dist = float('inf')
        
        for offset_y in range(-20, 20, 1):  
            x_temp, y_temp = x, y + offset_y
            while 0 <= x_temp < image.shape[1]:
                point = (float(x_temp), float(y_temp))
                dist = cv2.pointPolygonTest(contour, point, True)
                if dist >= 0 and abs(dist) < min_dist:
                    min_dist = abs(dist)
                    closest_point = (int(x_temp), target_y)  
                x_temp += step
        return closest_point

    left_point = find_nearest_contour_point(output_image, midfoot_pt, "left", max_contour, midfoot_pt[1])
    right_point = find_nearest_contour_point(output_image, midfoot_pt, "right", max_contour, midfoot_pt[1])

    if left_point and right_point:
        midfoot_width = math.dist(left_point, right_point) / pixels_per_cm
        draw_line_and_length(output_image, left_point, right_point, midfoot_width, (255, 0, 155), "Midfoot")
        cv2.circle(output_image, left_point, 12, (255, 0, 155), -1)  # 紫色圓點
        cv2.circle(output_image, right_point, 12, (255, 0, 155), -1)  # 紫色圓點

    # 後跟寬計算（黃線）
    heel_offset = int(0.15 * foot_length_pixels)
    heel_y = bottom_point[1] - heel_offset
    heel_center = (bottom_point[0], heel_y)

    heel_width_left, heel_width_right = find_nearest_contour_point(output_image, heel_center, "left", max_contour, heel_y), \
                                       find_nearest_contour_point(output_image, heel_center, "right", max_contour, heel_y)

    if heel_width_left and heel_width_right:
        heel_width = math.dist(heel_width_left, heel_width_right) / pixels_per_cm
        draw_line_and_length(output_image, heel_width_left, heel_width_right, heel_width, (0, 255, 255), "Heel")
        cv2.circle(output_image, heel_width_left, 12, (0, 255, 255), -1)  # 黃色圓點
        cv2.circle(output_image, heel_width_right, 12, (0, 255, 255), -1)  # 黃色圓點

   

    # 儲存處理後的圖片
    output_filename = os.path.join(OUTPUT_FOLDER, "result.png")
    cv2.imwrite(output_filename, output_image)

    elapsed_time = time.time() - start_time
    result_data = {
        "length_cm": round(insole_length_cm, 2),
        "forefoot_width_cm": round(forefoot_width, 2),
        "midfoot_width_cm": round(midfoot_width, 2),
        "heel_width_cm": round(heel_width, 2),
        "processing_time": round(elapsed_time, 2)
    }

    return "result.png", result_data

@app.route("/static/<filename>")
def get_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
