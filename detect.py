import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# buffer to smooth detection
hand_to_mouth_history = deque(maxlen=5)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands, \
     mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands
        hand_results = hands.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)

        hand_tip = None
        mouth_point = None

        # Get hand tip coordinates (index finger tip)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                hand_tip = np.array([int(hand_tip.x * w), int(hand_tip.y * h)])
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get mouth coordinates (average of lips)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            lip_upper = face_landmarks.landmark[13]  # upper lip
            lip_lower = face_landmarks.landmark[14]  # lower lip
            mouth_point = np.array([int((lip_upper.x + lip_lower.x)/2 * w),
                                    int((lip_upper.y + lip_lower.y)/2 * h)])

        # Check pill intake
        if hand_tip is not None and mouth_point is not None:
            distance = np.linalg.norm(hand_tip - mouth_point)
            hand_to_mouth_history.append(distance)

            # Detect intake if hand moves toward mouth
            if len(hand_to_mouth_history) == 5 and hand_to_mouth_history[-1] < 40:
                cv2.putText(frame, "Pill Intake Detected!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Pill Intake Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()


