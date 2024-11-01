import cv2
from ultralytics import YOLO

# Load the model
model = YOLO(r'C:\Users\lnwTutor\Desktop\DRYEYE D\best.pt')

# Define the class mapping
class_mapping = {
    0: "Dryeye",
    1: "Dryeye",
    2: "Eye_Normal"
}

# Open the webcam feed
cap = cv2.VideoCapture(0)
num_captures = 3
detections = []

print("Starting live webcam feed. Press 'c' to capture and analyze, 'q' to quit.")

capture_count = 0
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break

    # Display the live feed
    cv2.imshow("Webcam - Press 'c' to capture, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and capture_count < num_captures:
        print(f"Capturing and analyzing image {capture_count + 1}...")
        
        # Perform inference
        results = model(frame)
        
        # Process the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                # Use class mapping to determine the class label
                class_name = class_mapping.get(cls, "Unknown")
                print(f"Detected class: {class_name} with confidence {conf:.2f}")

                # Store the detection
                detections.append({"class": class_name, "confidence": conf})
                
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the captured frame with detection results
        cv2.imshow(f"Captured Image {capture_count + 1}", frame)
        capture_count += 1
    elif key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Calculate average confidence per class if detections were made
if detections:
    avg_confidence = {}
    for detection in detections:
        cls = detection["class"]
        conf = detection["confidence"]
        if cls in avg_confidence:
            avg_confidence[cls].append(conf)
        else:
            avg_confidence[cls] = [conf]

    # Output average confidence per class
    print("Detection Results and Average Confidence for Each Class:")
    for cls, confs in avg_confidence.items():
        avg_conf = sum(confs) / len(confs)
        print(f"{cls}: Average Confidence = {avg_conf:.2f}")
else:
    print("No detections were made.")
