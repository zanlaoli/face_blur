# coding:utf-8

import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image_file_name = '2'
image_file_name_suffix = 'jpg'
image = cv2.imread('%s.%s' % (image_file_name, image_file_name_suffix))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.02,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)

print('{0} results!'.format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('result', image)
file_name = '%s.%s' % (image_file_name + '_found', image_file_name_suffix)
cv2.imwrite(file_name, image)
cv2.waitKey(0)
