import time
import cv2
import os
import requests
from deepface import DeepFace
from audio_stt import SpeechToText
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_text_unicode_wrapped(frame, text, pos,
                              font_path="C:/Windows/Fonts/arial.ttf",
                              font_size=32, color=(255,255,0),
                              max_width_ratio=0.8):

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(font_path, font_size)

    img_w, img_h = img_pil.size
    max_width = int(img_w * max_width_ratio)

    def measure(text):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h

    words = text.split(" ")
    wrapped_lines = []
    current_line = ""

    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        w, _ = measure(test_line)

        if w <= max_width:
            current_line = test_line
        else:
            wrapped_lines.append(current_line)
            current_line = word

    if current_line:
        wrapped_lines.append(current_line)

    x, y = pos
    _, line_height = measure("A")
    line_height += 5

    for line in wrapped_lines:
        draw.text((x, y), line, font=font, fill=color)
        y += line_height

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

stt = SpeechToText("models/vosk-model-small-pt-0.3")
stt.start()

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

last_text = ""
last_time = 0
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

    text = stt.get_text()
    if text:
        last_text += text + " "
        last_time = time.time()

    if time.time() - last_time < 5 and last_text:
      frame = draw_text_unicode_wrapped(frame, f"Fala: {last_text}", (20, 70))

    else:
      last_text = ""
    cv2.imshow("Detector Facial - Base", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()