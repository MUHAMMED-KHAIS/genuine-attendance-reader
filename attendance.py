import cv2
import face_recognition
import numpy as np
import pyttsx3
import csv
from datetime import datetime

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed

# Load known face images
known_face_encodings = []
known_face_names = []

# Example: Load 2 people
person1_img = face_recognition.load_image_file(r"C:\Users\user\Desktop\opencv\known_faces\khais.jpg")
person1_encoding = face_recognition.face_encodings(person1_img)[0]
known_face_encodings.append(person1_encoding)
known_face_names.append("khais.png")

person2_img = face_recognition.load_image_file(r"C:\Users\user\Desktop\opencv\known_faces\khais.jpg")
person2_encoding = face_recognition.face_encodings(person2_img)[0]
known_face_encodings.append(person2_encoding)
known_face_names.append("Person 2")

# Initialize camera
video_capture = cv2.VideoCapture(0)

already_logged = set()  # To avoid repeating attendance
csv_filename = r"C:\Users\user\Desktop\opencv\unknown_attendance.csv"


# Create CSV file with header if not exists
try:
    with open(csv_filename, 'x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])
except FileExistsError:
    pass

while True:
    ret, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Scale back face location to original frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Mark attendance for unknown people
        if name == "Unknown" and name not in already_logged:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            with open(csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, date_str, time_str])

            engine.say("Unknown person detected")
            engine.runAndWait()

            already_logged.add(name)

    # Show the video
    cv2.imshow('Face Recognition', frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
