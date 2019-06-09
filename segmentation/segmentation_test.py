import numpy as np
import cv2
import math

bigkernel = np.ones((5,5),np.uint8)
smallkernel = np.ones((3,3),np.uint8)

image = cv2.imread("./images/sample1.tif")
image_copy = image.copy()

median = cv2.medianBlur(image, 5)
cv2.imshow("median", median)

gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

eq = cv2.equalizeHist(gray)
cv2.imshow("eq", eq)

ret, otsu = cv2.threshold(eq,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("otsu", otsu)

erode = cv2.erode(otsu, bigkernel, iterations = 1)
cv2.imshow("erode", erode)

(_, conts, _) = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

noobs = []
mask = np.zeros(gray.shape, dtype=np.uint8)

for c in conts:
  rRect = cv2.minAreaRect(c)
  rect = cv2.boundingRect(c)
  x, y = rect[0:2]
  width, height = rRect[1]
  if width == 0 or height == 0:
    continue

  area = width * height
  if area < 150:
    continue

  ratio = float(width) / float(height)
  if ratio < 2.5 and ratio > 0.4:
    continue

  noobs.append(c)

cv2.drawContours(mask, noobs, -1, (255,255,255), cv2.FILLED)

cv2.imshow("mask", mask)

# identify the background
otsudilate = cv2.dilate(otsu, bigkernel, iterations = 1)
cv2.imshow("otsudilate", otsudilate)

unknown = cv2.subtract(otsudilate, mask)
cv2.imshow("unknown", unknown)

ret, markers = cv2.connectedComponents(mask)
markers = markers + 1
markers[unknown==255] = 0

markers = cv2.watershed(median,markers)

mark = markers.astype('uint8')

mark = mark + 1
mark_transform = cv2.normalize(mark, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

cv2.imshow("mark_transform", mark_transform)

image_copy[markers == -1] = [0,255,0]
cv2.imshow("segmentation", image_copy)

cv2.waitKey(0)