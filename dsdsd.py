from cv2 import * #Import functions from OpenCV
import cv2
import numpy as np
def a():
    if __name__ == '__main__':
        kernel = np.ones((2,2), np.uint8) 
        kernel1 = np.ones((2,2), np.uint8)
        
        kernel2 = np.ones((2,2),np.float32)/5
        kernel3 = np.ones((2,2), np.uint8)
        source = cv2.imread('333.png')
        source = cv2.resize(source, (0,0), fx=3.5, fy=3.5)
        source = cv2.GaussianBlur(source,(5,5),0)
        source1 = cv2.dilate(source, kernel, iterations=1)
        #source2 = cv2.dilate(source, kernel1, iterations=1)
        #
        #denoise = cv2.fastNlMeansDenoising(source,None,255,255,255)
        closing = cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel)#####
        source2 = cv2.erode(source, kernel, iterations=1)
        #gradient = cv2.morphologyEx(source, cv2.MORPH_GRADIENT, kernel2)
        
        #blur = cv2.bilateralFilter(source,255,255,255)

          
        opening = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel2)
        #source = cv2.erode(source1, kernel3, iterations=1)
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
        #source = cv2.medianBlur(closing,3)
        median = cv2.medianBlur(source,5)
        
        #
        
        #source = cv2.dilate(source, kernel1, iterations=1)
        #opening = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel)
        final = cv2.medianBlur(np.float32(source),3)
        processed_image = cv2.filter2D(final,-1,kernel2)
        
        
        
        #blur1 = cv2.blur(source,(3,3))
        

        #final = cv2.dilate(final, kernel, iterations=1)
        #final = cv2.erode(final, kernel, iterations=1)
        #source = cv2.dilate(source, kernel, iterations=1)
        
        
         
        dst = cv2.filter2D(source,-1,kernel2)
        
        #cv2.imwrite('33.png', source)
        cv2.imshow('Source', source) #Show the image
        cv2.imshow('Source1', source1)
        cv2.imshow('Source2', source2) #Show the image
        cv2.imshow('Final', final) #Show the image
        cv2.imwrite('Final_Picture.png', source2)
        cv2.imshow('closing', closing)
        #cv2.imwrite('bnb.png', closing)
        #cv2.imshow('opening', opening)
        #cv2.imshow('median', median)
        #cv2.imshow('denoise', denoise)
        cv2.waitKey()

def b():
    if __name__ == '__main__':
        kernel = np.ones((2,2), np.uint8) 
        kernel1 = np.ones((2,2), np.uint8)
        
        kernel2 = np.ones((2,2),np.float32)/5
        kernel3 = np.ones((2,2), np.uint8)
        source = cv2.imread('hhh.png')
        source1 = cv2.dilate(source, kernel, iterations=1)
        source = cv2.GaussianBlur(source1,(5,5),1)
        
        source = cv2.resize(source1, (0,0), fx=2.5, fy=2.5)
        #source2 = cv2.dilate(source, kernel1, iterations=1)
        #
        #denoise = cv2.fastNlMeansDenoising(source,None,255,255,255)
        closing = cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel)#####
        source2 = cv2.erode(source, kernel, iterations=1)
        #gradient = cv2.morphologyEx(source, cv2.MORPH_GRADIENT, kernel2)
        
        #blur = cv2.bilateralFilter(source,255,255,255)

          
        opening = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel2)
        #source = cv2.erode(source1, kernel3, iterations=1)
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
        #source = cv2.medianBlur(closing,3)
        median = cv2.medianBlur(source,5)
        
        #
        
        #source = cv2.dilate(source, kernel1, iterations=1)
        #opening = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel)
        final = cv2.medianBlur(np.float32(source),3)
        processed_image = cv2.filter2D(final,-1,kernel2)
        
        
        
        #blur1 = cv2.blur(source,(3,3))
        

        #final = cv2.dilate(final, kernel, iterations=1)
        #final = cv2.erode(final, kernel, iterations=1)
        #source = cv2.dilate(source, kernel, iterations=1)
        
        
         
        dst = cv2.filter2D(source,-1,kernel2)
        
        #cv2.imwrite('33.png', source)
       #cv2.imshow('Source', source) #Show the image
        #cv2.imshow('Source1', source1)
        #cv2.imshow('Source2', source2) #Show the image
        #cv2.imshow('Final', final) #Show the image
        cv2.imwrite('Final_Picture.png', source2)
        cv2.imwrite('source1.png', source1)
        #cv2.imshow('closing', closing)
        #cv2.imwrite('bnb.png', closing)
        #cv2.imshow('opening', opening)
        #cv2.imshow('median', median)
        #cv2.imshow('denoise', denoise)
        #cv2.waitKey()
def c():
    kernel = np.ones((3,3), np.uint8)
    kernel1 = np.ones((4,4), np.uint8)
    kernel2 = np.ones((4,4), np.uint8)
    source = cv2.imread('source1.png')
    source1 = cv2.dilate(source, kernel, iterations=1)
    erosion = cv2.erode(source1,kernel,iterations = 1)
    
    
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    cv2.imwrite('source3.png', opening)
    

def d():
    #source = cv2.imread('333.png')
    #source = cv2.resize(source, (0,0), fx=4.5, fy=4.5)
    #cv2.imshow('closing', source)
    #cv2.waitKey()
    import cv2 
    from PIL import Image
    import numpy as np
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.morphology import watershed

    image = cv2.imread("333.jpg")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    darker = cv2.equalizeHist(gray)
    ret,thresh = cv2.threshold(darker,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    newimg = cv2.bitwise_not(thresh)

    im2, contours, hierarchy = cv2.findContours(newimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(newimg,[cnt],0,255,-1)

def e():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    img = cv2.imread('Final_Picture.png',0)
    img = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,3)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
def f():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    img = cv2.imread('Final_Picture.png',0)

    # global thresholding
    ret1,th1 = cv2.threshold(img,179,255,cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
              'Original Noisy Image','Histogram',"Otsu's Thresholding",
              'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    cv2.imwrite('source1.png', th3)

    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()
b()
f()
c()