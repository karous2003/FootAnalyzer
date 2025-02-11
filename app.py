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

    # A4 紙比例計算
    a4_width_cm = 21.0
    a4_length_cm = 29.7
    image_height, image_width = output_image.shape[:2]
    pixels_per_cm = ((image_width / a4_width_cm) + (image_height / a4_length_cm)) / 2

    # 找鞋墊長度
    bottom_point = tuple(max_contour[max_contour[:, :, 1].argmax()][0])
    front_point = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
    insole_length_cm = math.dist(bottom_point, front_point) / pixels_per_cm

    # 前掌寬 (紅線)
    def find_width_at_ratio(contour, ratio):
        y_target = int(bottom_point[1] - (bottom_point[1] - front_point[1]) * ratio)
        intersections = []
        for point in contour.squeeze():
            if abs(point[1] - y_target) < 5:
                intersections.append(point[0])
        if len(intersections) >= 2:
            return (min(intersections), y_target), (max(intersections), y_target), (max(intersections) - min(intersections)) / pixels_per_cm
        return None, None, 0

    forefoot_left, forefoot_right, forefoot_width = find_width_at_ratio(max_contour, 0.2)
    midfoot_left, midfoot_right, midfoot_width = find_width_at_ratio(max_contour, 0.5)
    heel_left, heel_right, heel_width = find_width_at_ratio(max_contour, 0.85)

    # 畫出主要測量點與線
    cv2.circle(output_image, bottom_point, 12, (255, 135, 0), -1)
    cv2.circle(output_image, front_point, 12, (255, 135, 0), -1)
    cv2.line(output_image, bottom_point, front_point, (255, 135, 0), 10)

    for pt_left, pt_right in [(forefoot_left, forefoot_right), (midfoot_left, midfoot_right), (heel_left, heel_right)]:
        if pt_left and pt_right:
            cv2.line(output_image, pt_left, pt_right, (0, 0, 255), 5)

    # 中足寬 (紫線)
    midfoot_y = int(bottom_point[1] - 0.4 * (bottom_point[1] - front_point[1]))
    midfoot_point = (bottom_point[0], midfoot_y)

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

    left_point = find_nearest_contour_point(output_image, midfoot_point, "left", max_contour, midfoot_point[1])
    right_point = find_nearest_contour_point(output_image, midfoot_point, "right", max_contour, midfoot_point[1])

    if left_point and right_point:
        cv2.circle(output_image, left_point, 12, (255, 0, 155), -1)  # 紫色圓點
        cv2.circle(output_image, right_point, 12, (255, 0, 155), -1)  # 紫色圓點
        cv2.line(output_image, left_point, right_point, (255, 0, 155), 5)  # 紫色線段
        midfoot_width_cm = math.dist(left_point, right_point) / pixels_per_cm
        mid_point = ((left_point[0] + right_point[0]) // 2, (left_point[1] + right_point[1]) // 2)
        cv2.putText(output_image, f"Midfoot Width: {midfoot_width_cm:.2f} cm", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 155), 2)
        print(f"中足寬 {midfoot_width_cm:.2f} cm")
    else:
        print("未找到中足的左右交點。")

    # 後跟寬 (黃線)
    heel_width_offset = 0.15 * (bottom_point[1] - front_point[1])
    heel_center_point_y = int(bottom_point[1] - heel_width_offset)
    heel_center_point = (bottom_point[0], heel_center_point_y)

    def find_heel_width_intersections(image, heel_center_point, contour):
        left_point = find_nearest_contour_point(image, heel_center_point, "left", contour, heel_center_point[1])
        right_point = find_nearest_contour_point(image, heel_center_point, "right", contour, heel_center_point[1])
        return left_point, right_point

    heel_width_left, heel_width_right = find_heel_width_intersections(output_image, heel_center_point, max_contour)

    if heel_width_left and heel_width_right:
        heel_width_cm = math.dist(heel_width_left, heel_width_right) / pixels_per_cm
        cv2.circle(output_image, heel_width_left, 12, (0, 255, 255), -1)  # 黃色圓點
        cv2.circle(output_image, heel_width_right, 12, (0, 255, 255), -1)  # 黃色圓點
        cv2.line(output_image, heel_width_left, heel_width_right, (0, 255, 255), 5)  # 黃色線段
        print(f"後跟寬 {heel_width_cm:.2f} cm")

        # 從後跟中心點直接畫到前掌中心點, 然後延長至碰到鞋墊邊緣
        green_line_start = heel_center_point
        green_line_end = forefoot_left  # 前掌中心點，可以根據需要調整
        line_direction = np.array([green_line_end[0] - green_line_start[0], green_line_end[1] - green_line_start[1]], dtype=float)
        line_length = np.linalg.norm(line_direction)
        line_direction /= line_length
        step_size = 15
        green_line_endpoint = None
        while True:
            green_line_end = (int(green_line_end[0] + line_direction[0] * step_size), int(green_line_end[1] + line_direction[1] * step_size))
            if cv2.pointPolygonTest(max_contour, green_line_end, True) < 0:
                green_line_endpoint = green_line_end
                break
        if green_line_endpoint is not None:
            cv2.circle(output_image, green_line_endpoint, 12, (0, 255, 0), -1)  # 綠色圓點
            cv2.circle(output_image, green_line_start, 12, (0, 255, 255), -1)  # 黃色圓點
            cv2.line(output_image, green_line_start, green_line_endpoint, (0, 255, 0), 5)  # 綠色線段
    else:
        print("未找到後跟寬度的左右交點。")

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
    
    # 保存結果圖片到 static 目錄
    output_filename = os.path.join('static', "result.png")
    cv2.imwrite(output_filename, output_image)

    return "result.png", result_data

@app.route("/static/<filename>")
def get_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  
