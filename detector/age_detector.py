import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt

from utils.functions import mod_box, find_center_imgs, tran
from utils.models import create_age_model, create_gender_model, create_emotion_model

agemodel = create_age_model()
genmodel = create_gender_model()

em_model = create_emotion_model()

em_model.load_weights('model/emotion_model.h5')
genmodel.load_weights('model/gen_model.h5')
agemodel.load_weights('model/age_model.h5')
# print('done')


def inference(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detections = detector.detect_faces(image)

    if len(detections) == 0:
        return image

    cropped_images = find_center_imgs(image, detections, 0.9)

    if len(cropped_images) == 0:
        return image
    ages = agemodel.predict(np.array(cropped_images)).astype(int)
    genders = genmodel.predict(np.array(cropped_images))
    genders = [0 if gen < 0.5 else 1 for gen in genders]

    emotion_imgs = [tran(img) for img in cropped_images]
    emotions = em_model.predict(np.array(emotion_imgs)).argmax(axis=1)


    classes = ['surprise', 'fear', 'sadness', 'disgust', 'contempt', 'happy', 'anger']

    gens = ["Male" if i == 0 else "Female" for i in genders]
    ems = [classes[i] for i in emotions]


    bboxes = [det['box'] for det in detections]

    for bbox, age, gen, em in zip(bboxes, ages, gens, ems):
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text1 = f'Age: {age[0]}'
        text3 = f'{em}'
        text2 = f'{gen}'

        y_text1 = y+h+20 if y-15 < 0 else y-15
        y_text3 = y+h+30 if y-5 < 0 else y-5
        color = (0, 0, 255)

        cv2.putText(image, text1, (x-5, y_text1), cv2.FONT_HERSHEY_SIMPLEX, 0.23, color, 1)
        cv2.putText(image, text3, (x-5, y_text3), cv2.FONT_HERSHEY_SIMPLEX, 0.23, color, 1)
        cv2.putText(image, text2, (x-5, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.23, color, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# img = cv2.cvtColor(cv2.imread('sad.jpeg'), cv2.COLOR_BGR2RGB)
# out = inference(img)
# plt.figure()
# plt.imshow(out)
# plt.show()

#
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    print(frame.shape)
    result_frame = inference(frame)

    # Display the result frame
    cv2.imshow('Face Detection and Analysis', cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


