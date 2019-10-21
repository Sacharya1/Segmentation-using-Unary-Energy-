# -*- coding: utf-8 -*-
"""
This code is highly inspired by online MATLAB code found on GitHub.
Link: https://github.com/akanazawa/MRF/blob/master/do_all.m 
@author: Sampad Acharya
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import matplotlib.image as mpimg
#Reading the Image

#I= cv2.imread('cow.jpg')
I=mpimg.imread('cow.jpg')
plot = plt.imshow(I)
plt.show()
#Converting grayscale image to RGB image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
   
Im = rgb2gray(I)

#Normalizing the image
Im=cv2.normalize(Im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

#Finding the neighbors for each pixel
dim= Im.shape   
row= dim[0]
col= dim[1]
r=row
c=col

numEl= row*col

def ind2sub(dim, ind):
    cols = math.floor(i/dim[1])
    rows = math.floor(i%dim[1])
    return rows, cols

def getNeighbors(i,r,c):
    rows, cols= ind2sub(dim, i)
    N=[]

    if rows<r-1:
        N.append(i+1)
    if rows>0:
        N.append(i-1)
    if cols<c-1:
        N.append(i+r)
    if cols>0:
        N.append(i-r)
    return N  
Neighbor=[]
for i in range(numEl):
    N= getNeighbors(i,r,c)
    Neighbor.append(N)

#Finading Unary energy

K= 0.897
uPrameters_sig= 30/255
uPrameters_mu_b= 200/255
uPrameters_mu_f1= 30/255
uPrameters_mu_f2= 120/255
pi= 3.14
Param=uPrameters_sig,uPrameters_mu_b,uPrameters_mu_f1,uPrameters_mu_f2


nodes=Im.reshape(numEl,1)


def unaryCost(nodes,Param):
    
    sig=Param[0]
    mu_b=Param[1]
    mu_f1=Param[2]
    mu_f2=Param[3]
    const= (1/2*math.log(2*pi))+ math.log(sig)
    alpha_b= (nodes- mu_b)**2/(2*sig**2) + const
    alpha_f = -np.log(np.exp(-(nodes - mu_f1)**2/(2*sig**2)) + np.exp(-(nodes - mu_f2)**2/(2*sig**2))) +const + np.log(2) 
    
    transAlpha_b= np.transpose(alpha_b)
    transAlpha_f= np.transpose(alpha_f)
    
    
    cost= np.concatenate((transAlpha_b,transAlpha_f ),axis=0)
    
    minInColumns = np.amin(cost, axis=0)
    print(minInColumns)
    argmax=np.argmin(cost, axis=0)
    argmax= argmax.reshape(1,numEl)
    lables= argmax
    Maximumlikihood= minInColumns
    U= np.sum(Maximumlikihood)

    return cost, U, lables
cost, U, lables=unaryCost(nodes,Param)

#Finding Unary image and plotting it on grayscale.
unaryImage= lables.reshape(row,col)

plot = plt.imshow(unaryImage)
                                                                                                                                                             
plot.set_cmap("gray")
