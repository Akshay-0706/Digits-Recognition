import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# initialize pygame
pygame.init()

x = 1200
y = 700
boundry = 5

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)

image = False
image_count = 1

bestmodel = load_model("bestmodel.h5")
predict = True
digits = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
          5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

isWriting = False
isErasing = False
x_arr = []
y_arr = []
# bg = DISPLAYSURF.mp_rgb(WHITE)
font = pygame.font.Font("Overpass.ttf", 18)
display = pygame.display.set_mode((x, y))
pygame.display.set_caption("Digits Recognition")

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and isWriting:
            x_cord, y_cord = event.pos
            pygame.draw.circle(display, white, (x_cord, y_cord), 6, 0)

            x_arr.append(x_cord)
            y_arr.append(y_cord)

        if isErasing:
            x_cord, y_cord = event.pos
            pygame.draw.circle(display, black, (x_cord, y_cord), 20, 0)

        if event.type == MOUSEBUTTONDOWN:
            if event.button == 3:
                isErasing = True
            if event.button == 1:
                isWriting = True

        if event.type == MOUSEBUTTONUP:
            if event.button == 3:
                isErasing = False
            if event.button == 1:
                isWriting = False
                x_arr = sorted(x_arr)
                y_arr = sorted(y_arr)
                rect_min_x, rect_max_x = max(
                    x_arr[0] - boundry, 0), min(x, x_arr[-1]+boundry)
                rect_min_y, rect_max_y = max(
                    0,  y_arr[0] - boundry), min(y_arr[-1]+boundry, x)

                x_arr = []
                y_arr = []

                img_arr = np.array(pygame.PixelArray(display))[
                    rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

                if image:
                    cv2.imwrite("image.png")
                    image_count += 1

                # predict the digit
                if predict:
                    predict_image = cv2.resize(img_arr, (28, 28))
                    predict_image = np.pad(
                        predict_image, (10, 10), 'constant', constant_values=0)
                    predict_image = cv2.resize(predict_image, (28, 28)) / 255

                    digit = str(digits[np.argmax(bestmodel.predict(
                        predict_image.reshape(1, 28, 28, 1)))])

                    text = font.render(digit, True, green, black)
                    text_rectangle = text.get_rect()
                    text_rectangle.left, text_rectangle.top = rect_min_x, rect_max_y

                    display.blit(text, text_rectangle)

        if event.type == KEYDOWN:
            if event.unicode == "c":
                display.fill(black)
        pygame.display.update()
