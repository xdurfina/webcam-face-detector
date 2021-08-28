import os
import pickle
import dlib
import cv2
import numpy as np
from shapely.geometry import Polygon


############################
# Made by Jaroslav Ďurfina #
############################


# Detecting face (Viola-Jones) and blurring background on webcam
def ViolaJones():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        xyz, img = cap.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (21, 21), 0)
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            img2 = img[y:y + h, x:x + w]
            blur[y:y + h, x:x + w] = img2
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('Viola-Jones Blur', blur)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

# Detecting face with DLIB (CNN) on webcam. Not great FPS, but it works.
def CNN():
    cap = cv2.VideoCapture(0)
    while True:
        xyz, image = cap.read()
        cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cnn_face_detector(gray, 1)

        for faceRect in faces:
            rect = faceRect.rect
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("CNN", image)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

# Scraping face data with Viola-Jones from 'data' folder
def getDataViolaJones(nop):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    dataViolaJones = [None] * nop

    for filename in os.listdir('data'):
        img = cv2.imread('data/' + filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        index = int(filename[:-4])
        dataViolaJones[index] = faces

    # Saving the data
    with open('dataViolaJones', 'wb') as fp:
        pickle.dump(dataViolaJones, fp)


# Scraping face data with DLIB(CNN) from 'data' folder
def getDataCNN(nop):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    dataCNN = [[]] * nop

    for filename in os.listdir('data'):
        image = cv2.imread('data/' + filename)
        # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        facesForThisPhoto = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cnn_face_detector(gray, 1)

        index = int(filename[:-4])

        for faceRect in faces:
            rect = faceRect.rect
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y

            face = [x, y, w, h]
            facesForThisPhoto.append(face)

        dataCNN[index] = facesForThisPhoto

    with open('dataCNN', 'wb') as fo:
        pickle.dump(dataCNN, fo)

# Loading Viola-Jones face data scraped from 'data' folder
def loadDataViolaJones():
    with open('dataViolaJones', 'rb') as fp:
        dataViolaJones = pickle.load(fp)
    for i in range(len(dataViolaJones)):
        if type(dataViolaJones[i]) == np.ndarray:
            dataViolaJones[i] = dataViolaJones[i].tolist()
        else:
            dataViolaJones[i] = []
    return dataViolaJones

# Loading DLIB(CNN) face data scraped from 'data' folder
def loadDataCNN():
    with open('dataCNN', 'rb') as fo:
        dataCNN = pickle.load(fo)
        return dataCNN


# Calculating intersection over union (IOU) of two rectangles
def calculate_iou(rectangle1, rectangle2):
    rectangle1 = [int(i) for i in rectangle1]
    rectangle2 = [int(i) for i in rectangle2]
    box_1 = [[rectangle1[0], rectangle1[1]], [rectangle1[0] + rectangle1[2], rectangle1[1]],
             [rectangle1[0] + rectangle1[2], rectangle1[1] + rectangle1[3]],
             [rectangle1[0], rectangle1[1] + rectangle1[3]]]
    box_2 = [[rectangle2[0], rectangle2[1]], [rectangle2[0] + rectangle2[2], rectangle2[1]],
             [rectangle2[0] + rectangle2[2], rectangle2[1] + rectangle2[3]],
             [rectangle2[0], rectangle2[1] + rectangle2[3]]]
    # X Y W H
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

# Printing rectangles
def PrintRectangles(i, data, color):
    for index, face in enumerate(data[i]):
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

# Change data to int format
def changeDataToIntFormat(data):
    for i, image in enumerate(data):
        for j, face in enumerate(image):
            for k, coordinate in enumerate(face):
                data[i][j][k] = int(coordinate)
    return data


if __name__ == "__main__":
    path, dirs, files = next(os.walk("data"))
    number_of_photos = len(files)

    ViolaJones()
    CNN()
    getDataViolaJones(number_of_photos)
    getDataCNN(number_of_photos)

    dataViolaJones = loadDataViolaJones()
    dataCNN = loadDataCNN()

    dataViolaJones = changeDataToIntFormat(dataViolaJones)
    dataCNN = changeDataToIntFormat(dataCNN)

    cv2.waitKey(0)

