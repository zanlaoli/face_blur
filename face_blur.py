# coding:utf-8

import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image_file_name = '1'
image_file_name_suffix = 'png'
image = cv2.imread('%s.%s' % (image_file_name, image_file_name_suffix))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.07,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)

print('{0} results!'.format(len(faces)))

for (x, y, w, h) in faces:
    image[y:y + w, x:x + h, :] = cv2.GaussianBlur(image[y:y + w, x:x + h, :],
                                                  (9, 9), 5)

cv2.imshow('result', image)
file_name = '%s.%s' % (image_file_name + '_blur', image_file_name_suffix)
cv2.imwrite(file_name, image)
cv2.waitKey(0)
