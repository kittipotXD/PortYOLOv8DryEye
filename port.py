import cv2
from ultralytics import YOLO

# โหลดโมเดล
model = YOLO(r'C:\Users\lnwTutor\Desktop\DRYEYE D\best.pt')

# กำหนดการแมพคลาส
class_mapping = {
    0: "Dryeye",
    1: "Dryeye",
    2: "Eye_Normal"
}

# เปิดฟีดกล้องเว็บแคม
cap = cv2.VideoCapture(0)
num_captures = 3
detections = []

print("เริ่มต้นฟีดกล้องเว็บแคม กด 'c' เพื่อจับภาพและวิเคราะห์, 'q' เพื่อออก")

capture_count = 0
while True:
    # จับภาพจากเว็บแคม
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถจับภาพจากเว็บแคมได้")
        break

    # แสดงฟีดสด
    cv2.imshow("กล้องเว็บแคม - กด 'c' เพื่อจับภาพ, 'q' เพื่อออก", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and capture_count < num_captures:
        print(f"จับภาพและวิเคราะห์ภาพที่ {capture_count + 1}...")

        # ทำการอนุมาน
        results = model(frame)

        # ประมวลผลผลลัพธ์
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                # ใช้การแมพคลาสเพื่อตรวจสอบชื่อคลาส
                class_name = class_mapping.get(cls, "Unknown")
                print(f"ตรวจพบคลาส: {class_name} ด้วยความมั่นใจ {conf:.2f}")

                # เก็บการตรวจจับ
                detections.append({"class": class_name, "confidence": conf})

                # วาดกรอบและป้ายกำกับบนเฟรม
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # แสดงเฟรมที่จับภาพพร้อมผลการตรวจจับ
        cv2.imshow(f"ภาพที่จับ {capture_count + 1}", frame)
        capture_count += 1
    elif key == ord('q'):
        break

# ปล่อยการจับภาพและปิดหน้าต่าง
cap.release()
cv2.destroyAllWindows()

# คำนวณความมั่นใจเฉลี่ยต่อคลาสถ้ามีการตรวจจับ
if detections:
    avg_confidence = {}
    for detection in detections:
        cls = detection["class"]
        conf = detection["confidence"]
        if cls in avg_confidence:
            avg_confidence[cls].append(conf)
        else:
            avg_confidence[cls] = [conf]

    # แสดงผลลัพธ์การตรวจจับและความมั่นใจเฉลี่ยต่อคลาส
    print("ผลลัพธ์การตรวจจับและความมั่นใจเฉลี่ยสำหรับแต่ละคลาส:")
    for cls, confs in avg_confidence.items():
        avg_conf = sum(confs) / len(confs)
        print(f"{cls}: ความมั่นใจเฉลี่ย = {avg_conf:.2f}")
else:
    print("ไม่มีการตรวจจับเกิดขึ้น")
