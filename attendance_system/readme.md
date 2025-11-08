
# Face Detection Attendance System

This project implements a face recognition-based attendance system using Python. It uses a webcam to detect faces, compares them against a database of known faces (stored as images), and logs attendance in a CSV file. The system is built with the `face_recognition` library and runs in a Python virtual environment.

## Project Structure
- `attendance.py`: Main script for face detection and attendance logging.
- `requirements.txt`: Lists all required Python packages with specific versions.
- `Attendance.csv`: CSV file to store attendance records (format: `Name,Time,Date`).
- `student_images/`: Directory containing JPG/PNG images of known faces (e.g., `FirstName_LastName.jpg`).

## Prerequisites
- **Operating System**: Windows (tested on Windows with PowerShell).
- **Python**: Version 3.12.3 (64-bit). Python 3.11 is recommended if compatibility issues arise.
- **Git**: Required for installing `face_recognition_models` from GitHub. Download from [git-scm.com](https://git-scm.com/download/win).
- **Webcam**: A working webcam (internal or external) for real-time face detection.
- **Image Files**: 2-3 clear, frontal-face JPG/PNG images (~200x200 pixels or larger) for the `student_images` directory.

## Setup Instructions
Follow these steps to set up and run the project.

### 1. Install Python
1. Download Python 3.12.3 (64-bit) from [python.org](https://www.python.org/downloads/release/python-3123/).
2. Install, ensuring **"Add Python to PATH"** is checked.
3. Verify installation:
   ```powershell
   python --version
   ```
   Expected: `Python 3.12.3`.

### 2. Clone or Create Project Directory
1. Create a project directory (e.g., `attendance_system`):
   ```powershell
   mkdir attendance_system
   cd attendance_system
   ```
2. If using a repository, clone it:
   ```powershell
   git clone <repository-url>
   cd attendance_system
   ```

### 3. Set Up Virtual Environment
1. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```
2. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   You should see `(venv)` in the PowerShell prompt.

### 4. Install Dependencies
1. Ensure `requirements.txt` is in the project directory with the following contents:
   ```
   click==8.3.0
   colorama==0.4.6
   dlib-bin==20.0.0
   face-recognition==1.3.0
   git+https://github.com/ageitgey/face_recognition_models#egg=face_recognition_models
   numpy==1.26.4
   opencv-python==4.10.0.84
   pillow==12.0.0
   setuptools<81
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Verify installed packages:
   ```powershell
   pip list
   ```
   Expected output includes:
   ```
   click                    8.3.0
   colorama                 0.4.6
   dlib-bin                 20.0.0
   face-recognition         1.3.0
   face_recognition_models  0.3.0
   numpy                    1.26.4
   opencv-python            4.10.0.84
   pillow                   12.0.0
   pip                      25.3
   setuptools               <81
   ```

### 5. Set Up Project Files
1. **Create Attendance.csv**:
   ```powershell
   echo Name,Time,Date > Attendance.csv
   ```
   This creates a CSV file with headers for attendance records.
2. **Create student_images Directory**:
   ```powershell
   mkdir student_images
   ```
3. **Add Images**:
   - Place 2-3 clear, frontal-face JPG/PNG images in `student_images` (e.g., `Somil_Jha.jpg`, `Alice_Smith.jpg`).
   - Images should be ~200x200 pixels or larger, with one face per image, named as `FirstName_LastName.jpg`.
4. **Update attendance.py**:
   - Ensure `attendance.py` is in the project directory. Copy the code below if needed.
   - Update the `os.environ["FACE_RECOGNITION_MODELS"]` path to match your virtual environment’s model directory (e.g., `C:\path\to\attendance_system\venv\Lib\site-packages\face_recognition_models\models`).

   ```python
   import cv2
   import face_recognition
   import os
   import numpy as np
   from datetime import datetime
   import face_recognition_models

   # Set explicit model path (update this path for your system)
   os.environ["FACE_RECOGNITION_MODELS"] = r"path\to\attendance_system\venv\Lib\site-packages\face_recognition_models\models"

   try:
       print("Starting script...")
       # Step 1: Define the path to student images (database)
       path = 'student_images'
       print(f"Checking directory: {path}")
       if not os.path.exists(path):
           raise FileNotFoundError(f"Directory {path} does not exist")
       
       images = []
       classNames = []
       
       # Load images and extract names
       myList = os.listdir(path)
       print(f"Files in {path}: {myList}")
       if not myList:
           raise FileNotFoundError(f"No files found in {path}")
       
       for cl in myList:
           curImg = cv2.imread(f'{path}/{cl}')
           if curImg is None:
               print(f"Failed to load image: {cl}")
               continue
           images.append(curImg)
           classNames.append(os.path.splitext(cl)[0])
       print("Loaded students:", classNames)
       
       if not images:
           raise ValueError("No valid images loaded for encoding")
       
       # Step 2: Function to generate face encodings from images
       def findEncodings(images):
           encodeList = []
           for i, img in enumerate(images):
               cl = myList[i]
               print(f"Encoding image: {cl}")
               img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
               encodings = face_recognition.face_encodings(img)
               if encodings:
                   encodeList.append(encodings[0])
               else:
                   print(f"No face found in image: {cl}")
           return encodeList

       encoded_face_train = findEncodings(images)
       print("Encodings complete.")
       
       if not encoded_face_train:
           raise ValueError("No face encodings generated")
       
       # Step 3: Function to mark attendance in CSV (only once per student)
       def markAttendance(name):
           print(f"Attempting to mark attendance for {name}")
           with open('Attendance.csv', 'r+') as f:
               myDataList = f.readlines()
               nameList = [line.split(',')[0] for line in myDataList]
               if name not in nameList:
                   now = datetime.now()
                   timeStr = now.strftime('%H:%M:%S')
                   dateStr = now.strftime('%d/%m/%Y')
                   f.writelines(f'\n{name},{timeStr},{dateStr}')
                   print(f"Attendance marked for {name}")

       # Step 4: Start webcam and perform real-time recognition
       print("Opening webcam...")
       cap = cv2.VideoCapture(0)  # 0 for default camera
       if not cap.isOpened():
           raise Exception("Could not open webcam")

       while True:
           success, img = cap.read()
           if not success:
               print("Failed to capture image.")
               break
           
           # Resize for faster processing
           imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
           imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
           
           # Find faces and encodings in current frame
           print("Detecting faces in frame...")
           facesCurFrame = face_recognition.face_locations(imgSmall)
           encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)
           
           for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
               matches = face_recognition.compare_faces(encoded_face_train, encodeFace)
               faceDis = face_recognition.face_distance(encoded_face_train, encodeFace)
               matchIndex = np.argmin(faceDis)
               
               if matches[matchIndex]:
                   name = classNames[matchIndex].upper()
                   # Scale locations back to original size
                   y1, x2, y2, x1 = faceLoc
                   y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                   
                   # Draw box and name (simple UI)
                   cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                   cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                   cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                   
                   # Mark attendance
                   markAttendance(name)
           
           # Display the video feed
           cv2.imshow('Attendance System', img)
           
           # Press 'q' to quit
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break

   except Exception as e:
       print(f"Error: {e}")
   finally:
       print("Cleaning up...")
       if 'cap' in locals():
           cap.release()
       cv2.destroyAllWindows()
   ```

### 6. Run the Project
1. Activate the virtual environment:
   ```powershell
   cd path\to\attendance_system
   .\venv\Scripts\Activate.ps1
   ```
2. Run the script:
   ```powershell
   python attendance.py
   ```
3. **Expected Output**:
   - Console:
     ```
     Starting script...
     Checking directory: student_images
     Files in student_images: ['Somil_Jha.jpg', 'Alice_Smith.jpg']
     Loaded students: ['Somil_Jha', 'Alice_Smith']
     Encoding image: Somil_Jha.jpg
     Encoding image: Alice_Smith.jpg
     Encodings complete.
     Opening webcam...
     Detecting faces in frame...
     Attendance marked for SOMIL_JHA
     ```
   - Webcam opens, showing green boxes around recognized faces.
   - `Attendance.csv` updated (e.g., `SOMIL_JHA,10:36:55,08/11/2025`).
   - Press 'q' to exit.

## Troubleshooting
Below are solutions to issues encountered during development.

### 1. `Please install face_recognition_models` Error
- **Symptoms**: Running `python -c "import face_recognition; print(face_recognition.__version__)"` or `attendance.py` fails with:
  ```
  Please install `face_recognition_models` with this command: pip install git+https://github.com/ageitgey/face_recognition_models
  ```
- **Cause**: `face_recognition` cannot detect `face_recognition_models`, despite being installed.
- **Solution**:
  1. Verify model files in `venv\Lib\site-packages\face_recognition_models\models`:
     ```powershell
     dir path\to\attendance_system\venv\Lib\site-packages\face_recognition_models\models
     ```
     Expected: `dlib_face_recognition_resnet_model_v1.dat`, `mmod_human_face_detector.dat`, `shape_predictor_5_face_landmarks.dat`, `shape_predictor_68_face_landmarks.dat`.
  2. Reinstall `face_recognition_models`:
     ```powershell
     pip uninstall face_recognition_models -y
     pip install git+https://github.com/ageitgey/face_recognition_models --no-cache-dir
     ```
  3. If it persists, try Python 3.11 (see below).

### 2. `ModuleNotFoundError: No module named 'pkg_resources'`
- **Symptoms**: Running scripts fails with:
  ```
  ModuleNotFoundError: No module named 'pkg_resources'
  ```
- **Cause**: `face_recognition_models` requires `pkg_resources` from `setuptools`, which is missing.
- **Solution**:
  - Install `setuptools`:
    ```powershell
    pip install "setuptools<81"
    ```
  - Verify:
    ```powershell
    pip list
    ```

### 3. `pkg_resources` Deprecation Warning
- **Symptoms**: Warning when running scripts:
  ```
  UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30.
  ```
- **Cause**: `face_recognition_models` uses the deprecated `pkg_resources` module.
- **Solution**:
  - The warning is benign for now. The `requirements.txt` pins `setuptools<81` to ensure compatibility.
  - After November 2025, consider switching to `deepface` or `mediapipe` for face recognition.

### 4. Python 3.12 Compatibility Issues
- **Symptoms**: Persistent `face_recognition_models` errors despite correct setup.
- **Cause**: `face_recognition==1.3.0` may be incompatible with Python 3.12.3.
- **Solution**:
  1. Install Python 3.11 from [python.org](https://www.python.org/downloads/release/python-3110/).
  2. Create a new virtual environment:
     ```powershell
     cd path\to\attendance_system
     python -m venv venv311
     .\venv311\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```
  3. Test:
     ```powershell
     python -c "import face_recognition; print(face_recognition.__version__)"
     ```

### 5. `Directory student_images does not exist` or `No files found in student_images`
- **Symptoms**: `attendance.py` fails with:
  ```
  Error: Directory student_images does not exist
  ```
  or
  ```
  Error: No files found in student_images
  ```
- **Solution**:
  - Create the directory and add images:
    ```powershell
    mkdir student_images
    dir path\to\attendance_system\student_images
    ```
  - Add 2-3 JPG/PNG images with clear, frontal faces (~200x200 pixels), named like `FirstName_LastName.jpg`.

### 6. `No face found in image` or `No valid images loaded for encoding`
- **Symptoms**: `attendance.py` fails with:
  ```
  Error: No face found in image: <image_name>
  ```
  or
  ```
  Error: No valid images loaded for encoding
  ```
- **Solution**:
  - Ensure `student_images` contains valid JPG/PNG images with one clear, frontal face each.
  - Verify image format and size (~200x200 pixels or larger).

### 7. Webcam Issues
- **Symptoms**: `attendance.py` fails with:
  ```
  Error: Could not open webcam
  ```
- **Solution**:
  - Test the webcam:
    ```python
    import cv2
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Test', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to access webcam")
    cap.release()
    ```
    Save as `test_camera.py`. Try `cv2.VideoCapture(1)` if it fails.
  - Ensure the webcam is connected and not used by another application.

### 8. Permission Errors for `Attendance.csv`
- **Symptoms**: `attendance.py` fails to write to `Attendance.csv`.
- **Solution**:
  - Ensure `Attendance.csv` is not open in another program.
  - Run as administrator:
    ```powershell
    Start-Process powershell -Verb runAs
    cd path\to\attendance_system
    python attendance.py
    ```

### 9. Slow Performance
- **Symptoms**: Face recognition is slow, causing delays in webcam processing.
- **Solution**:
  - In `attendance.py`, adjust the resize factor in the line:
    ```python
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    ```
    Change `0.25` to `0.5` for faster processing (less accuracy).

## Additional Notes
- **CMake Python Package**: Ensure no `cmake` Python package is installed:
  ```powershell
  pip uninstall cmake -y
  ```
- **Future Compatibility**: The `pkg_resources` deprecation may cause issues after November 2025. Monitor updates to `face_recognition` or consider alternatives like `deepface` or `mediapipe`.
- **Adding New Students**: Add new JPG/PNG images to `student_images` with clear, frontal faces, named as `FirstName_LastName.jpg`.

## Contact
For issues or enhancements, contact the project maintainer or open an issue in the repository (if applicable). Provide console output and error messages for faster resolution.

---

### Steps to Share
1. **Save README.md**:
   - Copy the above content into a file named `README.md` in `C:\Users\Somil\face\attendance_system`.
   - Use a text editor or PowerShell:
     ```powershell
     cd C:\Users\Somil\face\attendance_system
     echo <paste_content_above> > README.md
     ```

2. **Share Files**:
   - Share the following:
     - `README.md`
     - `requirements.txt`
     - `attendance.py`
     - `Attendance.csv` (optional, as it can be created)
     - `student_images` folder with sample images (if allowed, or instruct to add their own)
   - If using a Git repository, commit and push:
     ```powershell
     git init
     git add README.md requirements.txt attendance.py Attendance.csv student_images
     git commit -m "Initial commit of face detection attendance system"
     git remote add origin <repository-url>
     git push -u origin main
     ```

