import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model


# Define the model architecture
def create_emotion_model():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), input_shape=(48, 48, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='RMSprop')
    return model

def create_gender_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_age_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



em_model = load_model('detection/models/emotion_model.h5')
# genmodel = load_model('detection/models/gen_model.h5')

genmodel = create_gender_model()
genmodel.load_weights('detection/models/gen_model.h5')

# agemodel = load_model('detection/models/age_model.h5')
agemodel = create_age_model()
agemodel.load_weights('detection/models/age_model.h5')

def mod_box(box):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    if h > w:
        x = int(cx - h / 2)
        w = int(h)
    elif w > h:
        y = int(cy - w / 2)
        h = int(w)
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    return x, y, w, h

def find_center_imgs(img,detections,min_conf = 0.9):
    cropped_images = []
    for det in detections:
        if det['confidence'] >= min_conf:
            x, y, width, height = mod_box(det['box'])
            tem_img = img/255
            cropped_images.append(cv2.resize(tem_img[y:y+height,x:x+width,:], (200, 200)))
    return cropped_images

def tran(img):
    img = img * 255
    img = img.astype('uint8')
    resized_image = cv2.resize(img, (48, 48))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    final_image = np.expand_dims(gray_image, axis=-1)

    return final_image

def inference(image):
    # image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detections = detector.detect_faces(image)
    if len(detections) == 0:
        return image

    #     print(len(detections))
    cropped_images = find_center_imgs(image, detections, 0.9)
    if len(cropped_images) == 0:
        return image
    #     print(cropped_images)
    ages = agemodel.predict(np.array(cropped_images)).astype(int)
    genders = genmodel.predict(np.array(cropped_images))
    genders = [0 if gen < 0.5 else 1 for gen in genders]

    emotion_imgs = [tran(img) for img in cropped_images]
    emotions = em_model.predict(np.array(emotion_imgs)).argmax(axis=1)


    classes = ['surprise', 'fear', 'sadness', 'disgust', 'contempt', 'happy', 'anger']

    # plt.figure()

    gens = ["Male" if i == 0 else "Female" for i in genders]
    ems = [classes[i] for i in emotions]

    # print(ages)
    # print(["Male" if i == 0 else "Female" for i in genders])
    # print([classes[i] for i in emotions])

    bboxes = [det['box'] for det in detections]

    for bbox, age, gen, em in zip(bboxes, ages, gens, ems):
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text1 = f'Age: {age[0]}'
        text3 = f'{em}'
        text2 = f'{gen}'

        # y_text1 = max(y - 15, 15) y-15 if y-15
        # y_text3 = max(y - 5, 10)
        y_text1 = y+h+20 if y-15 < 0 else y-15
        y_text3 = y+h+30 if y-5 < 0 else y-5
        color = (0, 255, 0)

        cv2.putText(image, text1, (x-5, y_text1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(image, text3, (x-5, y_text3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(image, text2, (x-5, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image