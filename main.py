import cv2
import numpy as np
import math

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def roi(origin_binary):
    contours, _ = cv2.findContours(origin_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_squares = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour has 4 vertices (i.e., it is a square)
        if len(approx) == 4:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)

            # Check if the area is within a reasonable range for a square
            if area > 100:  # Adjust this threshold based on your image
                black_squares.append(approx)

    centers = []
    for square in black_squares:
        M = cv2.moments(square)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

    if not centers:
        return [-1,-1,-1,-1]
    centers = np.array(centers)
    x_min = int(np.min(centers[:, 0]))
    y_min = int(np.min(centers[:, 1]))
    x_max = int(np.max(centers[:, 0]))
    y_max = int(np.max(centers[:, 1]))
    return {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}


def read_origin(origin_path):
    correct_answers = []

    origin = cv2.imread(origin_path)
    origin_gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    _, origin_binary = cv2.threshold(origin_gray, 50, 255, cv2.THRESH_BINARY_INV)
    roi_result = roi(origin_binary)

    cv2.rectangle(origin, (roi_result['x_min'], roi_result['y_min']), (roi_result['x_max'], roi_result['y_max']), (0, 0, 255), 2)
    cropped_origin = origin[roi_result['y_min']:roi_result['y_max'], roi_result['x_min']:roi_result['x_max']]
    cropped_origin_binary = origin_binary[roi_result['y_min']:roi_result['y_max'], roi_result['x_min']:roi_result['x_max']]

    contours, _ = cv2.findContours(cropped_origin_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_dots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 440 < area < 480:
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if 0.4 <= circularity <= 0.9:
                white_dots.append(contour)

    for dot in white_dots:
        (x, y), radius = cv2.minEnclosingCircle(dot)
        center = (int(x), int(y))
        radius = int(radius) + 10
        correct_answers.append({'center': center, 'radius': radius})
        cv2.circle(cropped_origin, center, radius + 10, (255, 255, 0), 4)
    cv2.imwrite('origin_cropped.jpg', cropped_origin)

    return {'correct_answers': correct_answers, 'roi': roi_result}


def read_student(student_path):
    student_answers = []
    student = cv2.imread(student_path)
    student_gray = cv2.cvtColor(student, cv2.COLOR_BGR2GRAY)
    _, student_binary = cv2.threshold(student_gray, 50, 255, cv2.THRESH_BINARY_INV)
    roi_result = roi(student_binary)

    cv2.rectangle(student, (roi_result['x_min'], roi_result['y_min']), (roi_result['x_max'], roi_result['y_max']),
                  (0, 0, 255), 2)
    cropped_student = student[roi_result['y_min']:roi_result['y_max'], roi_result['x_min']:roi_result['x_max']]
    cropped_student_binary = student_binary[roi_result['y_min']:roi_result['y_max'],
                            roi_result['x_min']:roi_result['x_max']]

    contours, _ = cv2.findContours(cropped_student_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_dots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 440 < area < 480:
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if 0.4 <= circularity <= 0.9:
                white_dots.append(contour)

    for dot in white_dots:
        (x, y), radius = cv2.minEnclosingCircle(dot)
        center = (int(x), int(y))
        radius = int(radius) + 10
        student_answers.append({'center': center, 'radius': radius})
        cv2.circle(cropped_student, center, radius + 10, (255, 255, 0), 4)
    cv2.imwrite('student_cropped.jpg', cropped_student)

    return {'student_answers': student_answers, 'roi': roi_result}


def score(correct_answers,student_answers, roi_origin, roi_stundent, origin, student):

    origin = cv2.imread(origin)
    student = cv2.imread(student)
    cropped_origin = origin[roi_origin['y_min']:roi_origin['y_max'], roi_origin['x_min']:roi_origin['x_max']]
    cropped_student = student[roi_stundent['y_min']:roi_stundent['y_max'], roi_stundent['x_min']:roi_stundent['x_max']]
    difference = cv2.subtract(cropped_origin, cropped_student)
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(difference_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circle_count = 0
    for contour in contours:

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > 3:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            area = cv2.contourArea(contour)
            if 10 < area < 600:
                cv2.circle(difference, center, radius, (0, 255, 0), 4)
                circle_count += 1

    # Check if student circles are within any correct circle
    inside_occurrences = []
    for s_index, s_circle in enumerate(student_answers):
        for c_index, c_circle in enumerate(correct_answers):
            dist = distance(s_circle['center'], c_circle['center'])
            if dist <= c_circle['radius']:
                inside_occurrences.append((s_index, c_index))  # Save indices of student and correct circles

    print(f"Number of incorrect answers detected: {circle_count}")
    print(f"Number of correct answers detected: {len(inside_occurrences)}")

    if len(inside_occurrences) == len(correct_answers) - circle_count:
        return len(inside_occurrences)
    else:
        return max(len(inside_occurrences), len(correct_answers) - circle_count)



def main():
    correct_path = 'correct.jpg'
    student_path = 'student.jpg'
    origin_analyze = read_origin(correct_path)
    student_analyze = read_student(student_path)
    result = score(origin_analyze['correct_answers'], student_analyze['student_answers'], origin_analyze['roi'], student_analyze['roi'], correct_path, student_path)
    print(f"Number of correct answers: {result}")


if __name__ == "__main__":
    main()