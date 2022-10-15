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
from scipy import signal,ndimage

def rotateImage(image,angle,scale = 1):
	(h, w) = image.shape
	center = (int(w / 2) , int(h / 2)) 

	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotatedI = cv2.warpAffine(image, M, (w, h))
	return rotatedI

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
	#Generate two gaussian kernels with different K and Sigma values in a 7x7 matrix
	gaussianS1 = genGaussianKernel(7,1.6,1)
	gaussianS2 = genGaussianKernel(7,2,1.7)

	#X dir sobel filter
	Gx = np.array([[-1, 0, 1],[-2,0,2],[-1, 0, 1]])
	#Convolve and normalize gaussian kernel and sobel filter
	sobel1 = normalize(signal.convolve(gaussianS1,Gx),255)
	sobel2 = normalize(signal.convolve(gaussianS2,Gx),255)

	rotatedSobel = rotateImage(sobel1,90)

	filterBankS1 = sobel1
	filterBankS2 = sobel2
	#Rotate and concatenate images for filter
	for x in range(1,15):
		#Rotate both images
		rotatedS1 = rotateImage(sobel1,x*22.5)
		filterBankS1 = np.concatenate((filterBankS1,rotatedS1), axis=1)
		rotatedS2 = rotateImage(sobel2,x*22.5)
		filterBankS2 = np.concatenate((filterBankS2,rotatedS2), axis=1)


	DoGImage = np.concatenate((filterBankS1,filterBankS2), axis=0)
	DoG_imageWrite = cv2.imwrite('DoG.jpg',DoGImage)
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
 


