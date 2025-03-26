import os
import cv2
import numpy as np
import pygame
from joblib import load
from tensorflow.keras.models import load_model

# ✅ Get the absolute path of alert.wav
ALERT_SOUND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds", "alert.wav")

if not os.path.exists(ALERT_SOUND):
    print(f"❌ ERROR: Sound file not found at {ALERT_SOUND}")
else:
    print(f"✅ Sound file found: {ALERT_SOUND}")

# ✅ Initialize pygame mixer
pygame.mixer.init()

# ✅ Load trained models
svm_model = load("models/svm_drowsiness.joblib")
cnn_model = load_model("models/cnn_drowsiness.h5")

# ✅ Open webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        # ✅ CNN Prediction
        face_resized = cv2.resize(face, (224, 224)) / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)
        cnn_pred = cnn_model.predict(face_resized)[0][0]

        # ✅ SVM Prediction (Replace 0.3 with actual EAR calculation)
        svm_pred = svm_model.predict([[0.3]])[0]  

        print(f"CNN Prediction: {cnn_pred}, SVM Prediction: {svm_pred}")

        label = "Drowsy" if (cnn_pred > 0.5 or svm_pred == 1) else "Alert"
        color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)

        # ✅ Play sound alert when drowsy
        if label == "Drowsy" and os.path.exists(ALERT_SOUND):
            pygame.mixer.music.load(ALERT_SOUND)
            pygame.mixer.music.play()
        elif label == "Drowsy":
            print("❌ ERROR: alert.wav not found!")

        # ✅ Display result on webcam window
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
