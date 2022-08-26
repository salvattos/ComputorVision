#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:
import numpy as np
import cv2
import math
import os

def DoG(image):
	
	#gaussian filter
	sigma = 1.6
	K = 2
	gaussian = genGaussianKernel(7,1.6,1)

	cv2.imshow("image",cv2.resize(gaussian,(500,500)))
	cv2.waitKey(0)
	
	(h, w) = gaussian.shape
	threshhold = 100
	Gx = np.array([[-1, 0, 1],[-2,0,2],[-1, 0, 1]])
	Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
	sobel = np.zeros((h,w))
	#sobel filter
	for r in range(h-3):
		for c in range(w-3):
			S1 = np.sum(Gx*gaussian[r:r+3,c:c+3])
			S2 = np.sum(Gy*gaussian[r:r+3,c:c+3])
			S2 = 1
			sobel[r,c] = np.sqrt(pow(S1,2) + pow(S2,2))

	row,col = sobel.shape
	sobel = convolve2D(Gx,gaussian,5)
	normSobel = sobel.astype(np.uint8)	
	print(normSobel)
	cv2.imshow("image",cv2.resize(normSobel,(500,500)))
	cv2.waitKey(0)

	return gaussian

def genGaussianKernel(size,sigma,K=1):
	xmid = int(size/2)
	ymid = int(size/2)
	kernel = np.zeros((size,size))
	for x in range(0,size):
		for y in range(0,size):
			Z = .5*np.pi*pow(sigma,2)*pow(K,2)
			kernel[y,x] = Z*np.exp(-(pow(x-xmid,2)+pow(y-ymid,2))/(2*pow(K,2)*pow(sigma,2)))
	#normalize result
	normKernel = normalize(kernel,255)
	return normKernel

def normalize(array, upperBound):
	norm = (upperBound*(array - np.min(array))/np.ptp(array)).astype(np.uint8) 
	return norm

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def main():
	"""
		
	print("width={}, height={}, depth={}".format(w, h, d))
	# display the image to our screen -- we will need to click the window
	# open by OpenCV and press a key on our keyboard to continue execution
	cv2.imshow("Image", image)
	cv2.waitKey(0)
	(B, G, R) = image[100, 50]
	print("R={}, G={}, B={}".format(R, G, B))
	"""

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""

	directory  = "C:\\Users\\salva\\Documents\\WPI\\ComputerVision\\YourDirectoryID_hw0\\Phase1\\BSDS500\\Images"
	counter = 1

	image = cv2.imread("C:\\Users\\salva\\Documents\\WPI\\ComputerVision\\YourDirectoryID_hw0\\Phase1\\BSDS500\\Images\\1.jpg",0)
	DoGImage = DoG(image)
	#cv2.imshow("DoG",DoGImage)
	#cv2.waitKey(0)
	"""
	for filename in os.scandir(directory):
		if filename.is_file():
			print(filename.path)

		image = cv2.imread(filename.path,0)
		

		DoG_image = DoG(image)
		if(counter != 1):
			DoG_image = np.concatenate((oldDoG,DoG_image), axis=1)


		oldDoG = DoG_image
		#DoG_image = cv2.imwrite("test" + str(counter) + ".jpg",mag)
		counter += 1

	DoG_imageWrite = cv2.imwrite('DoG.jpg',DoG_image)
	"""
	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""



	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""


	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""


	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Brightness Map
	Perform brightness binning 
	"""


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Color Map
	Perform color binning or clustering
	"""


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
    
if __name__ == '__main__':
    main()
 


