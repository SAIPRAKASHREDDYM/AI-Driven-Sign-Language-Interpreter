import cv2
import numpy as np
import os
import time
from gtts import gTTS
from playsound import playsound
from tensorflow.keras.models import load_model

model = load_model("../model/sign_model.h5")

TRAIN_PATH = "../dataset/asl_alphabet_train"
labels = sorted(os.listdir(TRAIN_PATH))

IMG_SIZE = 64

last_spoken = ""
last_spoken_time = 0

# Pre-generate audio folder
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def speak(text):
    file_path = os.path.join(AUDIO_DIR, f"{text}.mp3")

    # Create audio file only once
    if not os.path.exists(file_path):
        tts = gTTS(text=text, lang="en")
        tts.save(file_path)

    playsound(file_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    x1, y1 = 50, 50
    x2, y2 = 350, 350

    roi = frame[y1:y2, x1:x2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi / 255.0
    roi = np.reshape(roi, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(roi, verbose=0)
    class_id = np.argmax(prediction)
    gesture = labels[class_id]
    confidence = float(np.max(prediction))

    current_time = time.time()

    if (
        confidence > 0.96
        and gesture != last_spoken
        and gesture not in ["nothing", "space", "del"]
        and (current_time - last_spoken_time) > 3
    ):
        print(f"Speaking: {gesture}  Confidence: {confidence:.2f}")
        speak(gesture)
        last_spoken = gesture
        last_spoken_time = current_time

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {gesture}", (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("AI Sign Language Recognition + Voice", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
