import cv2
import numpy as np

def detectar_componentes_conectados(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),10)

    ret, img_thr = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    countours, hierarchy = cv2.findContours(img_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(countour) for countour in countours]
    for rect in rectangles:
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)

    return img

def bordes(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img,100,200)
    return img

cam = cv2.VideoCapture(0)
while True:
    val, img = cam.read()
    img = detectar_componentes_conectados(img)
    cv2.imshow('Imagen',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
