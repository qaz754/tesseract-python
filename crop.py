import cv2
import pyautogui
from PIL import Image
import numpy as np



def a():
    im = Image.open('screenshot.png')
    data = np.array(im)

    r0, g0, b0 = 245, 233, 216  # Original value
    r1, g1, b1 = 216, 216, 216  # Original value
    r2, g2, b2 = 0, 0, 0  # Value that we want to replace it with

    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]

    im = Image.fromarray(data)
    im.save('screenshot.png')
def b():
    import cv2
    #import numpy as np

    img = cv2.imread('screenshot.png', 0)

    kernel = np.ones((2, 2), np.uint8)

    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    cv2.imshow('Input', img)
    cv2.imshow('Erosion', img_erosion)
    cv2.imshow('Dilation', img_dilation)

    cv2.waitKey(0)

def c():
    im = Image.open('screenshot.png')
    data = np.array(im)

    r1, g1, b1 = 245, 233, 216  # Original value
    r2, g2, b2 = 0, 0, 0  # Value that we want to replace it with

    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]

    im = Image.fromarray(data)
    im.save('screenshot.png')
def d():
    im = Image.open('screenshot.png')
    data = np.array(im)

    r1, g1, b1 = 240, 216, 216  # Original value
    r2, g2, b2 = 0, 0, 0  # Value that we want to replace it with

    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]

    im = Image.fromarray(data)
    im.save('screenshot.png')

def e():
    im = Image.open('screenshot.png')
    data = np.array(im)

    r1, g1, b1 = 216, 216, 234  # Original value
    r2, g2, b2 = 0, 0, 0  # Value that we want to replace it with

    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]

    im = Image.fromarray(data)
    im.save('screenshot.png')

def f():
    test_image1 = cv2.imread('a9.png')#Screenshot
    test_image = cv2.cvtColor(test_image1, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('abc.png',test_image)

def g():
    orig_color = (0, 0, 0)
    replacement_color = (255, 0, 0)
    img = Image.open('2080.png').convert('RGB')
    data = np.array(img)
    data[(data == orig_color).all(axis=-1)] = replacement_color
    img2 = Image.fromarray(data, mode='RGB')
    img2.save('screenshot.png')


    img2.show()

def h():
    list=[(127, 127, 127),(97, 97, 97),(111,111,111),[92,92,92]]#(216,240,249),(216, 233, 245),
    for x in list:
        orig_color = x
        replacement_color = (0, 0, 0)
        img = Image.open('dst.png').convert('RGB')
        data = np.array(img)
        data[(data == orig_color).all(axis=-1)] = replacement_color
        img2 = Image.fromarray(data, mode='RGB')
        img2.save('screenshot.png')
    
#a()
#c()
#d()
#e()
#f()
#g()
h()