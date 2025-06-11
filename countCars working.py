import cv2
import numpy as np
from time import sleep as delay

carNumber = 0
carPassed = set()
previousCar = 0
verificar = 0

cars = cv2.VideoCapture('project.mp4')

if not cars.isOpened():
    print("Error: Unable to open the video file.")
    exit()

while True:
    ret, frame = cars.read()
    if not ret:
        print("End of video or failed to read the frame.")
        break

    frame_resized = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    _, otsu_thresh = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    otsu_thresh_morph = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
    otsu_thresh_morph = cv2.morphologyEx(otsu_thresh_morph, cv2.MORPH_OPEN, kernel, iterations=5)

    otsu_thresh_colored = cv2.cvtColor(otsu_thresh_morph, cv2.COLOR_GRAY2BGR)

    min_height = min(frame_resized.shape[0], otsu_thresh_colored.shape[0])
    frame_resized = frame_resized[:min_height, :]
    otsu_thresh_colored = otsu_thresh_colored[:min_height, :]

    contours, _ = cv2.findContours(otsu_thresh_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lineOffset = 0.80
    line_y_upper = int(frame_resized.shape[0] * lineOffset - 6)
    line_y_lower = int(frame_resized.shape[0] * lineOffset + 12)
    cv2.line(frame_resized, (0, line_y_lower), (frame_resized.shape[1], line_y_lower), (176, 148, 49), 1)
    cv2.line(frame_resized, (0, line_y_upper), (frame_resized.shape[1], line_y_upper), (176, 148, 49), 1)
    tagIn = True
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 10000:
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.circle(frame_resized, (center_x, center_y), 1, (0, 255, 0), 1)

        if (center_y > line_y_upper) and (center_y < line_y_lower) and (tagIn == True) and (previousCar == carNumber):
                tagIn = False
                carNumber += 1

    cv2.putText(frame_resized,f'White cars: {carNumber}',(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    composed_frame = cv2.hconcat([frame_resized, otsu_thresh_colored])
    if (verificar >= 1):
        previousCar = carNumber
        verificar = 0
        tagIn = False
    if (previousCar != carNumber):
        verificar+=1
        tagIn = False
    
    cv2.imshow('Otsu Thresholding', composed_frame)
    
    delay(0.04)
    key = cv2.waitKey(0) & 0xFF  # 0 means wait indefinitely
    if key == ord('n'):
        continue
    elif key == 27:  # Escape key
        break

    if cv2.waitKey(1) & 0xFF == 27:  # Escape key
        break

cars.release()
cv2.destroyAllWindows()

    # Wait for 'n' key to move to the next frame
    

