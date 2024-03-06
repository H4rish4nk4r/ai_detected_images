

import psycopg2
from PIL import Image
from imgbeddings import imgbeddings

# File path of the face image
file_name = "solo-image.jpg"  # Replace with the path to your image

# Open the image
img = Image.open(file_name)

# Initialize the imgbeddings object
ibed = imgbeddings()

# Calculate the embeddings
embedding = ibed.to_embeddings(img)

# Connect to PostgreSQL database
conn = psycopg2.connect("postgres://avnadmin:AVNS_nHR6CTga0AQ2C57qKkK@pg-23f81deb-harirenjith123face.a.aivencloud.com:20445/defaultdb?sslmode=require")

# Create a cursor object
cur = conn.cursor()

# Convert embedding to a string representation for the query
string_representation = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"

# Execute the SQL query
cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))

# Fetch the results
rows = cur.fetchall()

# Close the cursor and connection
cur.close()
conn.close()

# Save the image to a specified location
save_path = "preview_image.jpg"  # Replace with the desired location and filename
img.save(save_path)

print(f"Image saved to {save_path}")
