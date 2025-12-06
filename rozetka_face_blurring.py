import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр с камеры.")
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            x_min = int(min([landmark.x for landmark in face_landmarks.landmark]) * w)
            x_max = int(max([landmark.x for landmark in face_landmarks.landmark]) * w)
            y_min = int(min([landmark.y for landmark in face_landmarks.landmark]) * h)
            y_max = int(max([landmark.y for landmark in face_landmarks.landmark]) * h)
            face_roi = image[y_min:y_max, x_min:x_max]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            image[y_min:y_max, x_min:x_max] = blurred_face
    cv2.imshow('MADE BY ROZETKA', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()