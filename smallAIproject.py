# train an AI model to do facial recognition on a raspberry pi
# using the camera module
# and the pi's GPIO pins

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import os
import RPi.GPIO as GPIO

# set up the GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(23, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# initialize the id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# loop over the frames from the video stream
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    # convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = detector.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )

    # loop through all the faces detected
    for(x,y,w,h) in faces:

        # create rectangle around the face
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

        # incrementing sample number
        id += 1

        # saving the captured face in the dataset folder
        cv2.imwrite("dataset/User." + str(id) + '.jpg', gray[y:y+h,x:x+w])

        # display the image
        cv2.imshow('image', image)

    # wait for 100 miliseconds
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video

    # if the sample number is more than 20
    if id > 20:
        break

# do a bit of cleanup
print(" [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

# train the model
os.system("python3 train.py")

# do a bit of cleanup
print(" [INFO] Exiting")
cam.release()
cv2.destroyAllWindows()

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# initialize the id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
    
# loop over the frames from the video stream
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    # convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = detector.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )

    # loop through all the faces detected
    for(x,y,w,h) in faces:

        # create rectangle around the face
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

        # recognize the face belongs to which id
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        # display the name
        cv2.putText(image, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(image, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    # display the image
    cv2.imshow('camera',image)

    # wait for 10 miliseconds
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video

    # if the sample number is more than 20
    if k == 27:
        break

# do a bit of cleanup
print(" [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

# train the model
os.system("python3 train.py")

# do a bit of cleanup
print(" [INFO] Exiting")
cam.release()
cv2.destroyAllWindows()

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# initialize the id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# loop over the frames from the video stream
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    # convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = detector.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )

