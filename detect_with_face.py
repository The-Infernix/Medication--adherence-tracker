from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np

# Load trained YOLO pill model
model = YOLO(r"runs/detect/train2/weights/best.pt")  # Update path if needed

# Open default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("✅ Camera started. Press 'q' to quit.")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Face pivot indices (for mouth and chin)
face_pivot_ids = [13, 14, 78, 308, 152, 1]  # upper lip, lower lip, corners, chin, nose tip

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands, \
     mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- YOLO pill detection ---
        results = model.predict(source=frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        pill_boxes = [(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                      for box in results[0].boxes.xyxy]

        # --- Hand & Face tracking ---
        hand_tip = None
        face_points = []

        hand_results = hands.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape

        # Hand pivot
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                hand_tip = np.array([int(tip.x * w), int(tip.y * h)])
                mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Face pivots
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            for idx in face_pivot_ids:
                lm = face_landmarks.landmark[idx]
                pt = np.array([int(lm.x * w), int(lm.y * h)])
                face_points.append(pt)
                cv2.circle(annotated_frame, (pt[0], pt[1]), 3, (0, 255, 0), -1)

            # Optional: calculate mouth center as pivot
            lip_upper = face_landmarks.landmark[13]
            lip_lower = face_landmarks.landmark[14]
            mouth_point = np.array([int((lip_upper.x + lip_lower.x)/2 * w),
                                    int((lip_upper.y + lip_lower.y)/2 * h)])
        else:
            mouth_point = None

        # --- Pill Intake Detection ---
        if hand_tip is not None and mouth_point is not None and pill_boxes:
            for (x1, y1, x2, y2) in pill_boxes:
                pill_center = np.array([(x1+x2)//2, (y1+y2)//2])
                distance_hand_pill = np.linalg.norm(hand_tip - pill_center)
                distance_pill_mouth = np.linalg.norm(pill_center - mouth_point)

                if distance_hand_pill < 50 and distance_pill_mouth < 100:
                    cv2.putText(annotated_frame, "Pill Intake Detected!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Pill Detection + Intake + Face Pivots", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera stopped.")
