from PIL import Image
import cv2
inPath = "file_0.png"
img = Image.open(inPath)
deg = 20
filterOpt = Image.BICUBIC
outPath = "Rotate_out.png"
foo = img.rotate(deg, filterOpt)
#cv2.imshow('Source2', foo) #S
foo.save(outPath)