import cv2
import face_recognition
import sqlite3
import pickle
import numpy as np
import csv
from datetime import datetime

def create_faces_table():
    # Connect to the SQLite database
    conn = sqlite3.connect("face_data.db")
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        encoding BLOB,
                        image BLOB)''')
    conn.commit()
    conn.close()

def log_recognition_to_csv(name, user_id):
    # Open the CSV file in append mode
    with open('recognition_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Write the log entry in the format: name, id, timestamp
        writer.writerow([name, user_id, timestamp])

def recognize_and_log_face():
    # Connect to the SQLite database
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

    last_recognized_name = None  # Variable to track the last recognized name

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Convert the image from BGR to RGB
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all faces and their encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the captured face encoding with known encodings
            distances = face_recognition.face_distance(known_encodings, face_encoding)

            # Find the best match and its distance
            best_match_index = np.argmin(distances)
            best_match_distance = distances[best_match_index]

            if best_match_distance < recognition_threshold:
                name = known_names[best_match_index]
                user_id = known_ids[best_match_index]
                print(f"Recognized {name} (ID: {user_id})")

                # Check if this name is different from the last recognized one to avoid duplicate entries
                if name != last_recognized_name:
                    # Log the recognition to the CSV file only when a new person is recognized
                    log_recognition_to_csv(name, user_id)
                    last_recognized_name = name  # Update the last recognized name
            else:
                name = "Unknown"
                user_id = None
                print("Face not recognized.")

            # Draw a rectangle around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Wait for the ESC key to exit the loop
        if cv2.waitKey(1) % 256 == 27:  # ESC key
            break

    # Release resources
    cam.release()
    cv2.destroyAllWindows()

    conn.close()

# Run the function
create_faces_table()  # Ensure the table exists
recognize_and_log_face()
