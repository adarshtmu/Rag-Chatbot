import os
from chromadb import Client

# Define the directory for the database
db_directory = "db"

# Ensure the directory exists
os.makedirs(db_directory, exist_ok=True)

# Initialize ChromaDB with the specified directory
client = Client(directory=db_directory)

# Example: Create a collection and add a document
collection = client.get_or_create_collection("my_collection")
collection.add({"id": "1", "content": "This is a test document."})

print(f"Database created inside '{db_directory}' directory.")
