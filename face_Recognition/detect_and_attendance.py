import cv2
import face_recognition
import sqlite3
import pickle
import numpy as np
import csv
from datetime import datetime

def load_face_data_from_db():
    # Connect to the SQLite database
    conn = sqlite3.connect("face_data.db")
    cursor = conn.cursor()

    # Get all face encodings and corresponding names from the database
    cursor.execute("SELECT id, name, encoding FROM faces")
    records = cursor.fetchall()

    # prepare a list of
    known_encodings = []
    known_names = []
    known_ids = []

    for record in records:
        user_id, name, encoding_blob = record
        # Deserialize the encoding from the database
        encoding = pickle.loads(encoding_blob)
        known_encodings.append(encoding)
        known_names.append(name)
        known_ids.append(user_id)

    conn.close()
    return known_encodings, known_names, known_ids

def log_recognition_to_csv(name, user_id):
    # Get the current time and date
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Open the CSV file in append mode
    with open("recognition_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        # Write the name, id, and timestamp to the CSV
        writer.writerow([name, user_id, current_time])

def recognize_and_log_face():
    # Load the existing face encodings, names, and IDs from the database
    known_encodings, known_names, known_ids = load_face_data_from_db()

    # Open the webcam
    cam = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cam.isOpened():
        print("Error: Could not access the webcam.")
        return

    cv2.namedWindow("Face Recognition")
    recognized = False  # Flag to ensure only one entry is made

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Convert the image from BGR to RGB (for face recognition)
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all faces and their encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the captured face encoding with the known encodings
            matches = face_recognition.compare_faces(known_encodings, face_encoding)

            name = "Unknown"  # Default name in case no match is found
            user_id = None  # Default ID if no match is found

            if True in matches:
                # Get the index of the first match
                first_match_index = matches.index(True)
                name = known_names[first_match_index]  # Get the name corresponding to the matched encoding
                user_id = known_ids[first_match_index]  # Get the ID corresponding to the matched encoding

                # Draw a rectangle around the face and put the name label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Only log the recognition once, when the user presses a specific key (e.g., spacebar)
                if not recognized:
                    log_recognition_to_csv(name, user_id)
                    recognized = True  # Mark as recognized, so no further entries are made

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Wait for ESC key to exit the loop or spacebar to log once
        key = cv2.waitKey(1) % 256
        if key == 27:  # ESC key
            break
        if key == 32:  # Spacebar to make the entry
            if recognized:
                print("Recognition logged.")
                break

    # Release resources
    cam.release()
    cv2.destroyAllWindows()

# Run the face recognition and logging function
recognize_and_log_face()
