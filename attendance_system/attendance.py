import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

try:
    # Step 1: Define the path to student images (database)
    path = 'student_images'
    images = []
    classNames = []

    # Load images and extract names
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is None:
            print(f"Failed to load image: {cl}")
            continue
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])  # Name without extension
    print("Loaded students:", classNames)

    # Step 2: Function to generate face encodings from images
    def findEncodings(images):
        encodeList = []
        for i, img in enumerate(images):
            cl = myList[i]  # Match with filename
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                encodeList.append(encodings[0])
            else:
                print(f"No face found in image: {cl}")
        return encodeList

    encoded_face_train = findEncodings(images)
    print("Encodings complete.")

    # Step 3: Function to mark attendance in CSV (only once per student)
    def markAttendance(name):
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
    cap = cv2.VideoCapture(0)  # 0 for default camera
    if not cap.isOpened():
        raise Exception("Could not open webcam")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            break
        
        imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
        
        facesCurFrame = face_recognition.face_locations(imgSmall)
        encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)
        
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encoded_face_train, encodeFace)
            faceDis = face_recognition.face_distance(encoded_face_train, encodeFace)
            matchIndex = np.argmin(faceDis)
            
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                
                # Draw box and name (simple UI)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
                
                markAttendance(name)

        cv2.imshow('Attendance System', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")
finally:

    cap.release()
    cv2.destroyAllWindows()