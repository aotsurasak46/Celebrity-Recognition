import cv2
import os 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for person in os.listdir('./Celebrity Faces Dataset'):
    print(person)
    for img in os.listdir(f'./Celebrity Faces Dataset/{person}'):
        path = f'./Celebrity Faces Dataset/{person}/{img}'

        image = cv2.imread(path)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]

            cropped_face = image[y:y + h, x:x + w]

            cv2.imwrite(f'./Celebrity Faces Dataset/{person}/{img}', cropped_face)
        else:
            print('No faces detected.')
            os.remove(f'./Celebrity Faces Dataset/{person}/{img}')

            