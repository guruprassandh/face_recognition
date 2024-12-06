import sqlite3

# Connect to SQLite database or create one
conn = sqlite3.connect("face_data.db")
cursor = conn.cursor()

# Create a table to store face encodings and images
cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    encoding BLOB NOT NULL,
    image BLOB NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()
