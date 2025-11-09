from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np

# Load your trained YOLO model
model = YOLO(r"runs/detect/pill_model_v83/weights/best.pt")  # <-- Update path if needed

# Open the default camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("✅ Camera started. Press 'q' to quit.")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands, \
     mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # mirror
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- YOLO pill detection ---
        results = model.predict(source=frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        pill_boxes = []
        for box in results[0].boxes.xyxy:  # x1, y1, x2, y2
            x1, y1, x2, y2 = map(int, box)
            pill_boxes.append((x1, y1, x2, y2))

        # --- Hand & Mouth tracking ---
        hand_tip = None
        mouth_point = None

        hand_results = hands.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                hand_tip = np.array([int(tip.x * w), int(tip.y * h)])
                mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            lip_upper = face_landmarks.landmark[13]
            lip_lower = face_landmarks.landmark[14]
            mouth_point = np.array([int((lip_upper.x + lip_lower.x)/2 * w),
                                    int((lip_upper.y + lip_lower.y)/2 * h)])

        # --- Pill Intake Detection ---
        if hand_tip is not None and mouth_point is not None and pill_boxes:
            for (x1, y1, x2, y2) in pill_boxes:
                pill_center = np.array([(x1+x2)//2, (y1+y2)//2])
                distance_hand_pill = np.linalg.norm(hand_tip - pill_center)
                distance_pill_mouth = np.linalg.norm(pill_center - mouth_point)

                if distance_hand_pill < 50 and distance_pill_mouth < 100:
                    cv2.putText(annotated_frame, "Pill Intake Detected!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Pill Detection + Intake", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera stopped.")
