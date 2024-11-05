import cv2
import pytesseract
import numpy as np

# Step 1: Read the image
image_path = 'origin.jpg'
image = cv2.imread(image_path)

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply thresholding to create a binary image
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Step 4: Use morphological operations to clean up the image
kernel = np.ones((2, 2), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Step 5: Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Extract the regions of interest (ROIs) corresponding to the letters
rois = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = binary[y:y+h, x:x+w]
    rois.append(roi)

# Step 7: Use Tesseract OCR to recognize the letters
recognized_texts = []
for roi in rois:
    custom_config = r'--oem 3 --psm 10'
    text = pytesseract.image_to_string(roi, config=custom_config)
    recognized_texts.append(text.strip())

# Step 8: Draw rectangles around the detected letters and display the recognized text
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, recognized_texts[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Step 9: Display the result
cv2.namedWindow('Detected Letters', cv2.WINDOW_NORMAL)
cv2.imshow('Detected Letters', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
