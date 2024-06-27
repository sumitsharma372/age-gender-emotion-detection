import cv2
import numpy as np

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



