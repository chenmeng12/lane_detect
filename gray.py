import cv2


image_path =  'test_images/test3.jpg'
image = cv2.imread(image_path)

blur = cv2.GaussianBlur(image, (5,5), 0)
edges = cv2.Canny(blur, 100, 150)

cv2.namedWindow('pic')
cv2.imshow('pic', image)
cv2.imshow('blur', blur)
cv2.imshow('edges', edges)
cv2.waitKey(0)