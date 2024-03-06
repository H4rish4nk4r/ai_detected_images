# importing the cv2 library
import cv2
import os

# loading the haar cascade algorithm file
alg = "haarcascade_frontalface_default.xml"
# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(alg)
# loading the image path
file_name = "image.jpg"
# reading the image
img = cv2.imread(file_name)
# detecting the faces
faces = haar_cascade.detectMultiScale(
    img, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100)
)

# check if the directory exists, if not, create it
output_dir = "stored-faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

i = 0
# for each face detected
for x, y, w, h in faces:
    # crop the image to select only the face
    cropped_image = img[y:y+h, x:x+w]
    # loading the target image path
    target_file_name = os.path.join(output_dir, f"{i}.jpg")
    cv2.imwrite(target_file_name, cropped_image)
    i += 1
