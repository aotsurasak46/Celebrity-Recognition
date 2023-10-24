import cv2
import numpy as np
import glob
import random
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os  # Import the os module to work with file paths

# Load the trained model
model = load_model("D:\\KMITL\\ImageProcess\\Project\\80mobilenetV2_bs8_e10_grey_lr0.0001_nn512_c17_frTrue.h5")

class_to_folder = {
    0: 'Angelina Jolie', 1: 'Brad Pitt', 2: 'Denzel Washington', 3: 'Hugh Jackman',
    4: 'Jennifer Lawrence', 5: 'Johnny Depp', 6: 'Kate Winslet', 7: 'Leonardo DiCaprio',
    8: 'Megan Fox', 9: 'Natalie Portman', 10: 'Nicole Kidman', 11: 'Robert Downey Jr',
    12: 'Sandra Bullock', 13: 'Scarlett Johansson', 14: 'Tom Cruise', 15: 'Tom Hanks', 16: 'Will Smith'
}

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def adj_detect_face(img, predicted_folder):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    rois = []
    for (x, y, w, h) in face_rects:
        roi = face_img[y:y+h,x:x+w]
        rois.append(roi)
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(face_img, f'Predicted: {predicted_folder}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return face_img, rois

# Open the video capture
cap = cv2.VideoCapture(1)

# Define the codec and create a VideoWriter object for saving the output
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('C:\\Users\\Jems\\Desktop\\DATA\\output.avi', fourcc, 20.0, (640, 480))

# Define the folder path where images of predicted celebrities are stored
# folder_path = 'C:\\Users\\Jems\\Desktop\\DATA\\dataset\\test\\'

# Create a new OpenCV display window
cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Recognition', 800, 600)

new_display_size = (180, 180)

predicted_folder = "Unknown"

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect faces in the frame
    detected_face, face_rois = adj_detect_face(frame, predicted_folder)

    # Make predictions on the face
    for roi in face_rois:
        image = cv2.resize(roi, (244, 244))
        image = img_to_array(image)
        image = image / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = np.expand_dims(image, axis=0)
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)
        predicted_folder = class_to_folder.get(predicted_class, "Unknown")

        # # Load an image from the folder specified by the predicted label
        # image_path = os.path.join(folder_path, predicted_folder, "sample.jpg")
        # image_to_display = load_img(image_path, target_size=(new_display_size[0], new_display_size[1]))

        # # Resize the loaded image to the new size
        # image_to_display = image_to_display.resize(new_display_size)

        # Overlay the new image on the video frame (in the top-left corner)
        detected_face[0:new_display_size[1], 0:new_display_size[0]] = img_to_array(image_to_display)[:,:,[2,1,0]]

    # Write the frame to the output video
    # out.write(detected_face)

    # Display the combined image in the new window
    cv2.imshow('Face Recognition', detected_face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()


