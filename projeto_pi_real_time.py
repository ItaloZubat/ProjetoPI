import cv2
import os
import requests
from deepface import DeepFace

if not os.path.exists("weights"):
    os.makedirs("weights")
    print("Pasta 'weights' criada.")


cascade_path = "weights/haarcascade_frontalface_default.xml"

if not os.path.isfile(cascade_path):
    print("Baixando Haarcascade...")

    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

    response = requests.get(url)
    if response.status_code == 200:
        with open(cascade_path, "wb") as f:
            f.write(response.content)
        print("Arquivo Haarcascade baixado com sucesso.")
    else:
        print("Erro ao baixar Haarcascade. Código:", response.status_code)
        exit()

face_detector = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir webcam.")
    exit()

print("Sistema iniciado! Pressione ESC para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        face_crop = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(
                face_crop,
                actions=['emotion'],
                enforce_detection=False
            )
            emotion = result[0]['dominant_emotion']
        except:
            emotion = "?"

        cv2.putText(frame, f"Emocao: {emotion}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    else:
        cv2.putText(frame, "Nenhum rosto detectado", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("Detector Facial - Base", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()