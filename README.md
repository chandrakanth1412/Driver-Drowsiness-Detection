# Driver Drowsiness Detection

A Windows webcam prototype that combines eye-state and yawning classifiers to
display drowsiness alerts, sound a local beep and write alert status to Firebase
Realtime Database.

## What is included

- Live OpenCV webcam detection with Haar face/eye cascades.
- Two VGG16-based classifiers for eye closure and yawning.
- Training scripts using a frozen ImageNet VGG16 base, a 256-unit dense layer,
  dropout and a two-class softmax output.
- Extracted eye/yawn training and validation/test datasets from the supplied archive.
- Three trained HDF5 model files, stored with Git LFS rather than ordinary Git blobs.
- Original cascade XML files and two supplied alarm audio files. The current live
  script uses `winsound.Beep`; it does not play those audio files.

## Repository layout

```text
code/
  live_drowsiness_detection.py
  firebase_test.py
  train_eye_model.py
  train_yawn_model.py
  eye_drowsiness_vgg16.h5
  eye_drowsiness_vgg16_v2.h5
  yawn_drowsiness_vgg16.h5
cascades/
eye_dataset/dataset/{train,val}/{closed,open}/
yawn_dataset/dataset/{train,test}/{no_yawn,yawn}/
```

## Setup on Windows

Install Python with a version supported by the TensorFlow release you choose,
Git and Git LFS. The original archive did not contain an environment lockfile;
`requirements.txt` records the direct imported libraries, not a verified version
matrix. Use an isolated virtual environment.

```powershell
git lfs install
git clone https://github.com/chandrakanth1412/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
git lfs pull
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The repository is private. Authenticate with the owning GitHub account to clone
and download its model files. Each model is about 130 MiB; verify that Git LFS
downloaded real model data rather than a small pointer file.

### Private Firebase configuration

Keep your service-account JSON outside the repository, or in the ignored
`code/firebase_key.json` path. Configure a Firebase Realtime Database that the
account may access. Never commit the service-account key or Android Firebase
configuration. Set these variables in the same PowerShell session:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\private\firebase_key.json"
$env:FIREBASE_DATABASE_URL = "https://YOUR-PROJECT-default-rtdb.REGION.firebasedatabase.app/"
.venv\Scripts\python.exe code/live_drowsiness_detection.py
```

The program opens webcam index 0. Press **Esc** in the OpenCV window to exit.
Firebase configuration is required by the current live implementation.
The supplied `.env.example` is a reference; the program does not load `.env` files.

`code/firebase_test.py` is a manual integration script: when explicitly run, it
writes `ON` to `alert/status`, waits five seconds and writes `OFF`. It is not an
offline unit test. No Android application source was present in the supplied ZIP.

## Training

```powershell
.venv\Scripts\python.exe code/train_eye_model.py
.venv\Scripts\python.exe code/train_yawn_model.py
```

Both scripts use 224 x 224 inputs, batch size 16 and ten epochs. Training may
download the ImageNet VGG16 weights on its first run. Eye training saves
`code/eye_drowsiness_vgg16_v2.h5`; live inference still uses the original
`code/eye_drowsiness_vgg16.h5`. Switch that path deliberately after validating
a replacement. Yawn training overwrites `code/yawn_drowsiness_vgg16.h5`.

## Current behavior and limitations

The eye threshold is more than ten closed-eye frames; yawning requires more than
25 frames with model confidence above 0.80. Frame-count thresholds depend on
actual inference speed. The prototype processes detected faces and only the
first detected eye for each face; it does not track identities across frames.

The five-second `winsound.Beep` call blocks the detection loop. The live script
writes Firebase `ON` when an alarm begins but does not currently write `OFF` when
the driver recovers. These existing behaviors were preserved during repository
preparation and are known follow-up work. No accuracy, real-time performance or
road-safety certification is claimed. This is a development prototype.

## Import and verification

Prepared from the supplied `project_driver.zip`. Application logic and original
model/data files were preserved. Absolute machine-specific paths were replaced
with project-relative paths; the Firebase URL and key location now use environment
configuration. Training outputs are saved under `code/`.

Private Firebase files and nested dataset archives were excluded. Their extracted
working datasets are included. The original ZIP remains unchanged on the owner's
machine. Python syntax and archive/file integrity were checked; webcam inference,
Firebase delivery and model training were not executed during the upload task.

## Dataset and asset provenance

The archive included datasets, trained weights, OpenCV cascade files and alarm
audio without a complete project-level provenance/licence record. Existing
notices inside cascade files are retained. No new licence or redistribution
permission is asserted for third-party data, pretrained weights or audio.
Complete that provenance record before making this repository public.
