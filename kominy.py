import cv2 as cv
import numpy as np


def resize_image(image, width):
    # Set the desired width
    desired_width = width
    w_percent = (desired_width / float(image.shape[1]))
    height = int((float(image.shape[0]) * float(w_percent)))
    dim = (desired_width, height)

    # resize image
    resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    return resized

def hough_p(img, edges):
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=30, maxLineGap=10)
    empty = np.ones_like(img) * 255
    if lines is None:
        return empty, img
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(empty, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return empty, img


def opening(img):
    kernel = np.ones((2, 1), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

def closing(img):
    kernel = np.ones((3, 1), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

def kmeans(img):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    resx = center[label.flatten()]
    res2 = resx.reshape(img.shape)
    cv.imshow('kmeans', res2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return res2

img = cv.imread('chimn8.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img = resize_image(img, 500)

opened = opening(img)
opened = opening(opened)
blur = cv.medianBlur(opened, 3, 0)
blur = cv.equalizeHist(blur)
print("shape:", blur.shape)
clustered = kmeans(blur)
cv.imshow('opening', blur)
edges = cv.Canny(clustered, 210, 240)
_, res = hough_p(img, edges)
opened = opening(edges)
#closed = closing(opened)


cv.imshow('canny', edges)
#cv.imshow('opened', opened)
cv.imshow('hough', res)
cv.waitKey(0)
cv.destroyAllWindows()

