from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import firebase_admin
from firebase_admin import credentials, db
import time

# -------------------------------
# INITIALIZE FIREBASE
# -------------------------------
cred = credentials.Certificate(
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", str(PROJECT_ROOT / "code" / "firebase_key.json"))
)

firebase_admin.initialize_app(cred, {
    "databaseURL": os.environ["FIREBASE_DATABASE_URL"]
})

# -------------------------------
# FIREBASE REFERENCE
# -------------------------------
alert_ref = db.reference("alert/status")

# -------------------------------
# TEST: TURN ALERT ON
# -------------------------------
print("Turning ALERT ON")
alert_ref.set("ON")
time.sleep(5)

# -------------------------------
# TEST: TURN ALERT OFF
# -------------------------------
print("Turning ALERT OFF")
alert_ref.set("OFF")

print("Test completed")
