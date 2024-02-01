import math
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
from matplotlib import pyplot as plt
import cv2

def prepocess_image(img_path:str):
    image = Image.open(img_path)
    image = image.resize((416, 416))
    image = image.convert("RGB")
    image = torch.tensor(image).unsqueeze(0)

    return image

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, labels, scores):
    for box, label, score in zip(boxes, labels, scores):
        x, y, w, h = box
        color = (255, 0, 255)  # Green color for bounding boxes
        label_text = f"{label}: {score:.2f}"

        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.putText(image, label_text, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Open the default camera (index 0)
cam = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()

    # Perform object recognition or any other processing on the frame here
    # image = prepocess_image(frame)
    inputs = image_processor(images=frame, return_tensors="pt")

    outputs = model(**inputs)
    target_sizes = torch.tensor([frame.shape[:2]])

    results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        draw_boxes(frame, results["boxes"], results["labels"], results["scores"])

    # Display the frame
    cv2.imshow("YOLO", frame)

    # Add a delay and check for a key press
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' key is pressed
    if key == ord('q'):
        break

# Release the camera and close the window
cam.release()
cv2.destroyAllWindows()

