import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = r"C:\OBJECT DETECTION\Pill Training\onnx_model\best_model.onnx"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g., [8, 3, 640, 640]
expected_batch = input_shape[0] if isinstance(input_shape[0], int) else 1

def preprocess(img, size=640):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    img = padded.transpose((2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32) / 255.0
    return img, scale, (w, h)

def postprocess(outputs, scale, orig_shape, conf_thres=0.4):
    preds = outputs[0]

    # handle case: (batch, num_boxes, num_attrs)
    if len(preds.shape) == 3:
        preds = preds[0]  # take first batch

    boxes, scores = [], []
    for det in preds:
        # det = [x1, y1, x2, y2, conf, cls1, cls2, ...]
        if det[4] < conf_thres:
            continue

        x1, y1, x2, y2 = det[:4] / scale
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        scores.append(float(det[4]))

    return boxes, scores


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor, scale, orig_shape = preprocess(frame)

    # repeat image to match expected batch size
    input_tensor = np.repeat(input_tensor, expected_batch, axis=0)

    outputs = session.run(None, {input_name: input_tensor})

    boxes, scores = postprocess(outputs, scale, orig_shape)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Pill {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Pill Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
