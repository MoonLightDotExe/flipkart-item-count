import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolo11n.pt")  # Use YOLOv8 model for initial detection

# Load the input image
image_path = "path_to_your_image_2.png"  # Replace with your image path
image = cv2.imread(image_path)

# Perform object detection
results = model.predict(image, conf=0.25)

# Extract bounding boxes and masks
detections = results[0].boxes.data.cpu().numpy() if len(results[0].boxes) > 0 else []
masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

# Convert image to grayscale for Watershed
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocess: Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Increased kernel size for more smoothing

# Adaptive thresholding for better binarization
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# Morphological operations for noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)

# Background and Foreground separation
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# Adjusted threshold multiplier for distance transform
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Mark unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling for Watershed
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Apply Watershed algorithm
markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]  # Mark boundaries in red

# Count unique segments in the Watershed result
unique_segments = len(np.unique(markers)) - 2  # Exclude background and boundary

# Function to check if an item is inside a bounding box
def is_inside_bbox(x1, y1, x2, y2, markers):
    y_indices, x_indices = np.where(markers > 0)
    return any((x1 <= x <= x2 and y1 <= y <= y2) for x, y in zip(x_indices, y_indices))

# Count items inside YOLO bounding boxes
count_inside_bbox = 0
for det in detections:
    x1, y1, x2, y2, conf, cls = det
    if is_inside_bbox(x1, y1, x2, y2, markers):
        count_inside_bbox += 1
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Display the results
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Total Items Detected: {count_inside_bbox}")
plt.axis('off')
plt.show()

print(f"Total number of items detected: {count_inside_bbox}")
