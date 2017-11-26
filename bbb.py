import cv2 #This will give an error if you don't have cv2 module
img = cv2.imread('333.png') #bat.jpg is the batman image.
#make sure that you have saved it in the same folder
img = cv2.resize(img, (0,0), fx=2, fy=2)
#Averaging
avging = cv2.blur(img,(4,4)) #You can change the kernel size as you want
cv2.imshow('Averaging',avging)
cv2.waitKey(0)

#Gaussian Blurring
gausBlur = cv2.GaussianBlur(img, (3,3),0) #Again, you can change the kernel size
cv2.imshow('Gaussian Blurring', gausBlur)
cv2.waitKey(0)

#Median blurring
medBlur = cv2.medianBlur(img,5)
cv2.imshow('Media Blurring', medBlur)
cv2.waitKey(0)

#Bilateral Filtering
bilFilter = cv2.bilateralFilter(img,0,0,0)
cv2.imshow('Bilateral Filtering', bilFilter)
cv2.waitKey(0)
cv2.destroyAllWindows()
