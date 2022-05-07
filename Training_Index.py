from tkinter import *
from PIL import Image, ImageTk

import os
import time

from tensorflow import keras

import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


##############################################
#                GAME                        #
##############################################

class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = 0, 0  # previous head point

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.imgFood = cv2.resize(self.imgFood, (0, 0), None, 0.3, 0.3)

        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 600), random.randint(100, 400)

    def update(self, imgMain, currentHead):

        if self.gameOver == True:
            cvzone.putTextRect(imgMain, "Game Over", [100, 200], scale=2, thickness=5, offset=20)
            cvzone.putTextRect(imgMain, f'Puntaje: {self.score}', [200, 350], scale=2, thickness=5, offset=20)
        else:
            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # Length Reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLength:
                        break

            # Check if snake ate the Food
            rx, ry = self.foodPoint
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score += 1
                print(self.score)

            # Draw Snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
                cv2.circle(imgMain, self.points[-1], 20, (0, 255, 0), cv2.FILLED)

            # Draw Food
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood,[rx - self.wFood // 2 , ry - self.hFood // 2 ])

            cvzone.putTextRect(imgMain, f'Puntaje: {self.score}', [50, 80], scale=3, thickness=3, offset=10)

            # Check for Collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 255, 0), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

            print(minDist)

            if -1 <= minDist <= 1:
                print("Hit")
                self.gameOver = True
                self.points = []  # all points of the snake
                self.lengths = []  # distance between each point
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 150  # total allowed Length
                self.previousHead = 0, 0  # previous head point
                self.randomFoodLocation()

        return imgMain


game = SnakeGameClass("images/Donut.png")

mode_game = False
##############################################
##############################################



#############################################
frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL


mode_asl = True
model = keras.models.load_model('model_train.h5')

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if classNo == 0:
        return 'A'
    elif classNo == 1:
        return 'B'
    elif classNo == 2:
        return 'C'
    elif classNo == 3:
        return 'D'

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

folderPath = "ex_images"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

start_time = time.time()
terminate_time = time.time()
imgNum = random.randrange(0, 4)
assertPoint = 0
imgAnterior = imgNum



root = Tk()
root.title("Entrenador ASL")
root.geometry("1280x607")

root.resizable(0,0)


f1 = LabelFrame(root,bg="black")
f1.pack(side=RIGHT,padx=35)

L1 = Label(f1,bg="black")
L1.pack()


background = PhotoImage(file="images/portal-copia.png")
back = Label(root, image=background)
back.pack()

def close():
    root.destroy()

nivel_time = 0
nivel_text = ""

terminateNivelTime = 0

def resetAssertPoint():
    global assertPoint
    assertPoint = 0

def resetTerminateTime():
    global terminate_time
    terminate_time = time.time()

nivel_number = 0

def nivel1():
    global mode_asl
    mode_asl = True
    global mode_game
    mode_game = False
    global nivel_number
    nivel_number = 1
    global nivel_time
    nivel_time = 20
    global nivel_text
    nivel_text = "Nivel 1"
    global terminateNivelTime
    terminateNivelTime = 90
    resetTerminateTime()
    resetAssertPoint()

def nivel2():
    global mode_asl
    mode_asl = True
    global mode_game
    mode_game = False
    global nivel_number
    nivel_number = 2
    global nivel_time
    nivel_time = 10
    global nivel_text
    nivel_text = "Nivel 2"
    global terminateNivelTime
    terminateNivelTime = 60
    resetTerminateTime()
    resetAssertPoint()

def nivel3():
    global mode_asl
    mode_asl = True
    global mode_game
    mode_game = False
    global nivel_number
    nivel_number = 3
    global nivel_time
    nivel_time = 5
    global nivel_text
    nivel_text = "Nivel 3"
    global terminateNivelTime
    terminateNivelTime = 40
    resetTerminateTime()
    resetAssertPoint()

def newDialog():
    infoWin = Tk()
    infoWin.title("Información")
    infoWin.geometry("100x100")
    infoWin.resizable(0, 0)

def startGame():
    global mode_asl
    mode_asl = False
    global mode_game
    mode_game = True
    game.gameOver = False
    game.score = 0

def get_ayuda():
    global background
    background = PhotoImage(file="images/Ayuda.png")
    back.configure(image=background)
    button_h.configure(text="Quitar ayuda", command=get_init)
    #background = PhotoImage(file="images/Ayuda.png")
    #backHelp = Label(root, image=background)
    #backHelp.pack()

def get_init():
    global background
    background = PhotoImage(file="images/portal-copia.png")
    back.configure(image=background)
    button_h.configure(text="Ayuda (?)", command=get_ayuda)

cap = cv2.VideoCapture(0)


label = Label(text="Hello, Tkinter", background="#34A2FE")
label.pack()


btn_nivel1 = Button(root,text="Nivel 1", bg="#F7E79C",relief="flat", height=2, width=10, command=nivel1)
btn_nivel1.place(x = 100, y = 550)

btn_nivel2 = Button(root,text="Nivel 2", bg="#F7E79C",relief="flat", height=2, width=10, command=nivel2)
btn_nivel2.place(x = 240, y = 550)

btn_nivel3 = Button(root,text="Nivel 3", bg="#F7E79C",relief="flat", height=2, width=10, command=nivel3)
btn_nivel3.place(x = 380, y = 550)


button = Button(root,text="Salir!", bg="#F7E79C",relief="flat", height=2, width=10, command=close)
button.place(x = 1100, y = 550)

button_h = Button(root,text="Ayuda (?)", bg="#F7E79C",relief="flat", height=2, width=10, command=get_ayuda)
button_h.place(x = 980, y = 550)

button_g = Button(root,text="Juego", bg="#F7E79C",relief="flat", height=2, width=10, command=startGame)
button_g.place(x = 780, y = 550)



while True:
      img1 = cap.read()[1]

      if mode_asl == True:

        # READ IMAGE
        success, imgOrignal = cap.read()
        imgOrignal = cv2.flip(imgOrignal, 1)
        # PROCESS IMAGE
        img = np.asarray(imgOrignal)
        handsImg, img = detector.findHands(imgOrignal)

        current_time = time.time()

        if nivel_time == 0 and nivel_number > 0:
            if nivel_number == 1:
                if assertPoint > 18:
                    menssage = "Felicidades eres un experto, "
                    menssage2 = "tuviste mas de 18 aciertos!!! :)"
                elif assertPoint == 18:
                    menssage = "Felicidades tuviste 18 aciertos, "
                    menssage2 = "casi eres un experto, sigue asi!!! :)"
                elif assertPoint < 18:
                    menssage = "Lo siento, tuviste menos de 18 aciertos, "
                    menssage2 = "debes seguir practicando :("
            elif nivel_number == 2:
                if assertPoint > 10:
                    menssage = "Felicidades eres un experto, "
                    menssage2 = "tuviste mas de 10 aciertos!!! :)"
                elif assertPoint == 10:
                    menssage = "Felicidades tuviste 10 aciertos, "
                    menssage2 = "casi eres un experto, sigue asi!!! :)"
                elif assertPoint < 10:
                     menssage = "Lo siento,tuviste menos de 10 aciertos,"
                     menssage2 = " debes seguir practicando :("
            elif nivel_number == 3:
                if assertPoint > 5:
                      menssage = "Felicidades eres un experto, "
                      menssage2 = " tuviste mas de 5 aciertos!!! :)"
                elif assertPoint == 5:
                     menssage = "Felicidades tuviste 5 aciertos,"
                     menssage2 = "casi eres un experto, sigue asi!!! :)"
                elif assertPoint < 5:
                      menssage = "Lo siento,tuviste menos de 5 aciertos, "
                      menssage2 = " debes seguir practicando :("
            cv2.putText(imgOrignal, menssage, (20, 240), font, 0.75, (0, 0, 255), 2, cv2.LINE_4)
            cv2.putText(imgOrignal, menssage2, (20, 270), font, 0.75, (0, 0, 255), 2, cv2.LINE_4)


        if nivel_time > 0:
            if current_time - terminate_time > terminateNivelTime:
                nivel_time = 0

            if current_time - start_time > nivel_time:
                imgNum = random.randrange(0, 4)
                start_time = time.time()

            h, w, c = overlayList[imgNum].shape
            img[0:h, 0:w] = overlayList[imgNum]

            cv2.rectangle(img, (0, 300), (200, 640), (213, 245, 227), cv2.FILLED)
            cv2.putText(imgOrignal, str(getCalssName(imgNum)), (45, 435), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 20)

            cv2.putText(imgOrignal, nivel_text, (63, 470), font, 0.75, (255, 0, 0), 2, cv2.LINE_4)

            cv2.putText(imgOrignal, "Puntos: ", (500, 35), font, 0.75, (255, 0, 0), 2, cv2.LINE_4)
            cv2.putText(imgOrignal, str(assertPoint), (599, 35), font, 0.75, (255, 0, 0), 2, cv2.LINE_4)

            if len(handsImg) != 0:

                img = cv2.resize(img, (200, 200))
                img = preprocessing(img)
                # cv2.imshow("Processed Image", img)
                img = img.reshape(1, 200, 200, 1)

                #cv2.putText(imgOrignal, "MATCH: ", (220, 40), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

                # PREDICT IMAGE
                predictions = model.predict(img)
                # classIndex =  model.predict_classes(img)
                classIndex = np.argmax(predictions, axis=1)
                probabilityValue = np.amax(predictions)

                if probabilityValue > threshold:

                    if ((round(probabilityValue * 100, 2)) >= 99 and (getCalssName(classIndex) == getCalssName(imgNum))):

                        cv2.putText(imgOrignal, str(getCalssName(classIndex)), (45, 435), cv2.FONT_HERSHEY_PLAIN, 10,
                                    (255, 0, 0), 20)
                        # cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (55, 460), font, 0.75,
                        #             (255, 0, 0),
                        #             2, cv2.LINE_AA)
                        assertPoint = assertPoint + 1
                        #cv2.imshow("Result", imgOrignal)

                        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)

                        imgNum = random.randrange(0, 4)
                        start_time = time.time()

        #cv2.imshow("Result", imgOrignal)

        img1 = cv2.cvtColor(imgOrignal, cv2.COLOR_BGRA2RGB)
        img1 = ImageTk.PhotoImage(Image.fromarray(img1))
        L1['image'] = img1



        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

      elif mode_game == True:

          success, img = cap.read()
          img = cv2.flip(img, 1)
          hands, img = detector.findHands(img, flipType=False)

          if hands:
              lmList = hands[0]['lmList']
              pointIndex = lmList[8][0:2]
              img = game.update(img, pointIndex)
          #cv2.imshow("Image", img)

          img4 = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
          img4 = ImageTk.PhotoImage(Image.fromarray(img4))
          L1['image'] = img4

          key = cv2.waitKey(1)
          if key == ord('r'):
              game.gameOver = False

      root.update()

cap.release()



