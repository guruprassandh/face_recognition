import cv2
import face_recognition
import sqlite3
import pickle
import numpy as np
import csv
from datetime import datetime

# Create faces table if it doesn't exist
def create_faces_table():
    conn = sqlite3.connect("face_data.db")
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        encoding BLOB,
                        image BLOB)''')
    conn.commit()
    conn.close()

# Log the recognition to CSV file once key is pressed
def log_recognition_to_csv(name, user_id):
    with open('recognition_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, user_id, timestamp])

# Recognize faces and log when confirmation is given
def recognize_and_log_face():
    conn = sqlite3.connect("face_data.db")
    cursor = conn.cursor()

    # Load stored face encodings and names
    cursor.execute("SELECT id, name, encoding FROM faces")
    records = cursor.fetchall()

    known_encodings = []
    known_names = []
    known_ids = []

    for record in records:
        user_id, name, encoding_blob = record
        encoding = pickle.loads(encoding_blob)
        if isinstance(encoding, np.ndarray):  # Ensure it's a valid NumPy array
            known_encodings.append(encoding)
            known_names.append(name)
            known_ids.append(user_id)

    # Open the webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not access the webcam.")
        return

    cv2.namedWindow("Face Recognition")
    recognition_threshold = 0.6  # Confidence threshold
    recognized_ids = set()  # To track recognized IDs in the current session
    last_recognition_time = {}  # Dictionary to store the last recognition time for each ID

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            best_match_distance = distances[best_match_index]

            if best_match_distance < recognition_threshold:
                name = known_names[best_match_index]
                user_id = known_ids[best_match_index]
                current_time = datetime.now()

                # Check if the individual has been recognized recently (within a threshold time window)
                if user_id not in recognized_ids or (current_time - last_recognition_time.get(user_id, current_time)).seconds > 3:
                    print(f"Recognized {name} (ID: {user_id})")

                    # Display the face rectangle and name
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Wait for user confirmation (key press)
                    key = cv2.waitKey(0)  # Wait for any key press
                    if key == 13:  # Enter key to log
                        log_recognition_to_csv(name, user_id)
                        recognized_ids.add(user_id)  # Mark this ID as recognized
                        last_recognition_time[user_id] = current_time  # Update the time of recognition

                else:
                    print(f"Duplicate recognition of {name}, skipping logging...")

            else:
                name = "Unknown"
                user_id = None
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) % 256 == 27:  # ESC key
            break

    cam.release()
    cv2.destroyAllWindows()
    conn.close()

# Run the face recognition system
create_faces_table()  # Ensure the table exists
recognize_and_log_face()
