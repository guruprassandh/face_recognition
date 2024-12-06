import cv2
import face_recognition
import sqlite3
import pickle

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

def capture_and_store_face():
    # Connect to the SQLite database
    conn = sqlite3.connect("face_data.db")
    cursor = conn.cursor()

    # Get user input for name and ID
    name = input("Enter the name of the person: ")
    user_id = input("Enter the ID of the person: ")

    # Open the webcam
    cam = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cam.isOpened():
        print("Error: Could not access the webcam.")
        return

    cv2.namedWindow("Capture Face")
    print("Position your face in front of the camera to start capturing.")
    encoding_list = []  # List to store multiple encodings
    sample_count = 0  # Count of captured samples

    while sample_count < 5:  # Capture 5 samples per person
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Display the frame
        cv2.imshow("Capture Face", frame)

        # Process the frame for face recognition
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)

        if encodings:
            # Take the first encoding (if multiple faces are detected, it handles only the first one)
            encoding_list.append(encodings[0])
            sample_count += 1
            print(f"Captured {sample_count} samples.")

        # Wait for the ESC key to exit the loop
        if cv2.waitKey(1) % 256 == 27:  # ESC key
            print("Face capture aborted.")
            break

    # Serialize the encodings for storage
    encoding_blob = pickle.dumps(encoding_list)

    # Convert the captured frame (image) to a byte array (using the last captured frame)
    _, img_encoded = cv2.imencode('.jpg', frame)
    image_blob = img_encoded.tobytes()

    # Insert the data into the database
    try:
        cursor.execute("INSERT INTO faces (id, name, encoding, image) VALUES (?, ?, ?, ?)", 
                       (user_id, name, encoding_blob, image_blob)) 
        conn.commit()
        print(f"Face data for {name} (ID: {user_id}) stored successfully!")
    except sqlite3.Error as e:
        print(f"Error inserting data into the database: {e}")

    # Release resources
    cam.release()
    cv2.destroyAllWindows()
    conn.close()

# Run the function
create_faces_table()  # Ensure the table exists
capture_and_store_face()
