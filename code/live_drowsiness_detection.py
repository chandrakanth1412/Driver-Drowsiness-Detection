from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import cv2
import numpy as np
import winsound   # ✅ REQUIRED FOR BEEP
from tensorflow.keras.models import load_model

# 🔥 FIREBASE IMPORTS (ADDED)
import firebase_admin
from firebase_admin import credentials, db

# -------------------------------
# FIREBASE INITIALIZATION (ADDED)
# -------------------------------
cred = credentials.Certificate(
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", str(PROJECT_ROOT / "code" / "firebase_key.json"))
)

firebase_admin.initialize_app(cred, {
    "databaseURL": os.environ["FIREBASE_DATABASE_URL"]
})

alert_ref = db.reference("alert/status")

# -------------------------------
# LOAD HAAR CASCADES
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    str(PROJECT_ROOT / 'cascades' / 'haarcascade_frontalface_default.xml')
)

eye_cascade = cv2.CascadeClassifier(
    str(PROJECT_ROOT / 'cascades' / 'haarcascade_eye.xml')
)

# -------------------------------
# LOAD EYE MODEL
# -------------------------------
eye_model = load_model(
    str(PROJECT_ROOT / 'code' / 'eye_drowsiness_vgg16.h5')
)

eye_labels = ["closed", "open"]

# -------------------------------
# LOAD YAWNING MODEL
# -------------------------------
yawn_model = load_model(
    str(PROJECT_ROOT / 'code' / 'yawn_drowsiness_vgg16.h5')
)

yawn_labels = ["no_yawn", "yawn"]

# -------------------------------
# DROWSINESS PARAMETERS (EYES)
# -------------------------------
EYE_CLOSED_FRAMES = 0
EYE_CLOSED_THRESHOLD = 10 

# -------------------------------
# DROWSINESS PARAMETERS (YAWN)
# -------------------------------
YAWN_FRAMES = 0
YAWN_THRESHOLD = 25         # ~2–3 seconds
LAST_YAWN_LABEL = "no_yawn"

# -------------------------------
# ALARM STATE
# -------------------------------
ALARM_ON = False

# -------------------------------
# OPEN WEBCAM
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not detected")
    exit()

print("✅ Webcam started successfully")

# -------------------------------
# LIVE LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # -------------------------------
        # EYE DETECTION
        # -------------------------------
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            EYE_CLOSED_FRAMES = 0

        for (ex, ey, ew, eh) in eyes[:1]:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]

            if eye_img.size == 0:
                continue

            eye_img = cv2.resize(eye_img, (224, 224))
            eye_img = eye_img / 255.0
            eye_img = np.reshape(eye_img, (1, 224, 224, 3))

            prediction = eye_model.predict(eye_img, verbose=0)
            label = eye_labels[np.argmax(prediction)]

            if label == "closed":
                EYE_CLOSED_FRAMES += 1
            else:
                EYE_CLOSED_FRAMES = 0

            color = (0, 255, 0) if label == "open" else (0, 0, 255)

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                color,
                2
            )

            cv2.putText(
                roi_color,
                label,
                (ex, ey - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # -------------------------------
        # MOUTH REGION (ROI)
        # -------------------------------
        mouth_y_start = int(h * 0.6)
        mouth_y_end = h

        mouth_x_start = int(w * 0.2)
        mouth_x_end = int(w * 0.8)

        mouth_roi = roi_color[
            mouth_y_start:mouth_y_end,
            mouth_x_start:mouth_x_end
        ]

        cv2.rectangle(
            roi_color,
            (mouth_x_start, mouth_y_start),
            (mouth_x_end, mouth_y_end),
            (0, 255, 255),
            2
        )

        # -------------------------------
        # YAWNING PREDICTION
        # -------------------------------
        if mouth_roi.size != 0:
            mouth_img = cv2.resize(mouth_roi, (224, 224))
            mouth_img = mouth_img / 255.0
            mouth_img = np.reshape(mouth_img, (1, 224, 224, 3))

            yawn_pred = yawn_model.predict(mouth_img, verbose=0)
            yawn_confidence = np.max(yawn_pred)

            if yawn_confidence > 0.80:
                yawn_label = yawn_labels[np.argmax(yawn_pred)]
            else:
                yawn_label = "no_yawn"

            LAST_YAWN_LABEL = yawn_label

            if yawn_label == "yawn":
                YAWN_FRAMES += 1
            else:
                YAWN_FRAMES = 0

            cv2.putText(
                roi_color,
                yawn_label,
                (mouth_x_start, mouth_y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # -------------------------------
        # COMBINED DROWSINESS DECISION
        # -------------------------------
        DROWSY = False
        DROWSY_REASON = ""

        if EYE_CLOSED_FRAMES > EYE_CLOSED_THRESHOLD:
            DROWSY = True
            DROWSY_REASON = "EYES CLOSED"

        if YAWN_FRAMES > YAWN_THRESHOLD:
            DROWSY = True
            DROWSY_REASON = "YAWNING"

        # -------------------------------
        # DISPLAY ALERT + BEEP + FIREBASE
        # -------------------------------
        if DROWSY:
            cv2.putText(
                frame,
                f"Reason: {DROWSY_REASON}",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            if not ALARM_ON:
                winsound.Beep(1000, 5000)   # PC beep
                alert_ref.set("ON")          # 🔥 APK ALERT
                ALARM_ON = True
        else:
            ALARM_ON = False

    cv2.imshow("Driver Monitor - Eye & Yawn Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
