import torch
import torchvision.transforms as transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
import cv2
from PIL import Image
import numpy as np


# Load the CPU-only version of the model


# Initialize the DETR processor and model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-cpu")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50-cpu")

# Set up the video capture (0 for default camera, or specify a video file)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam or video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a format suitable for inference
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Parse the detection results
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    # Visualize the detected objects on the frame
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw bounding boxes and labels on the frame
        class_name = model.config.id2label[label.item()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name}: {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detected objects
    cv2.imshow('Object Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
