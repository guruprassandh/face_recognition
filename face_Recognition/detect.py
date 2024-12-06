import cv2
import face_recognition
import sqlite3
import pickle
import numpy as np

def load_face_data_from_db():
    # Connect to the SQLite database
    conn = sqlite3.connect("face_data.db")
    cursor = conn.cursor()

    # Get all face encodings and corresponding names from the database
    cursor.execute("SELECT id, name, encoding FROM faces")
    records = cursor.fetchall()

    # Prepare a list of known encodings and names
    known_encodings = []
    known_names = []

    for record in records:
        user_id, name, encoding_blob = record
        # Deserialize the encoding from the database
        encoding = pickle.loads(encoding_blob)
        known_encodings.append(encoding)
        known_names.append(name)

    conn.close()
    return known_encodings, known_names

def recognize_face():
    # Load the existing face encodings and names from the database
    known_encodings, known_names = load_face_data_from_db()

    # Open the webcam
    cam = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cam.isOpened():
        print("Error: Could not access the webcam.")
        return

    cv2.namedWindow("Face Recognition")

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
            if True in matches:
                # Get the index of the first match
                first_match_index = matches.index(True)
                name = known_names[first_match_index]  # Get the name corresponding to the matched encoding

            # Draw a rectangle around the face and put the name label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Wait for ESC key to exit the loop
        if cv2.waitKey(1) % 256 == 27:  # ESC key
            break

    # Release resources
    cam.release()
    cv2.destroyAllWindows()

# Run the face recognition function
recognize_face()
