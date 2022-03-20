# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:31:35 2022

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

source1 = np.load("CNN Source Values Code Test Run Data1.npy", allow_pickle=True)
source2 = np.load("CNN Source Values Code Test Run Data2.npy", allow_pickle=True)
source3 = np.load("CNN Source Values Code Test Run Data3.npy", allow_pickle=True)
source4 = np.load("CNN Source Values Code Test Run Data4.npy", allow_pickle=True)
source5 = np.load("CNN Source Values Code Test Run Data5.npy", allow_pickle=True)


xSC = []
ySC = []

xML = []
yML = []

xNC = []
yNC = []

xLab = []
yLab = []

LossXValues = []

for i in range(200):
    LossXValues.append(i)


for i in range(17):
    x = sum(source1[i][3])/len(source1[i][3]) + sum(source2[i][3])/len(source2[i][3]) + sum(source3[i][3])/len(source3[i][3]) + sum(source4[i][3])/len(source4[i][3]) + sum(source5[i][3])/len(source5[i][3])
        
    xSC.append(source1[i][1])
    ySC.append(x*0.2)
        
   
for i in range(17, 34):
    x = sum(source1[i][3])/len(source1[i][3]) + sum(source2[i][3])/len(source2[i][3]) + sum(source3[i][3])/len(source3[i][3]) + sum(source4[i][3])/len(source4[i][3]) + sum(source5[i][3])/len(source5[i][3])
        
    xML.append(source1[i][1])
    yML.append(x*0.2)
    
for i in range(34, 51):
    x = sum(source1[i][3])/len(source1[i][3]) + sum(source2[i][3])/len(source2[i][3]) + sum(source3[i][3])/len(source3[i][3]) + sum(source4[i][3])/len(source4[i][3]) + sum(source5[i][3])/len(source5[i][3])
        
    xNC.append(source1[i][1])
    yNC.append(x*0.2)  
    
    
for i in range(51, 68):
    x = sum(source1[i][3])/len(source1[i][3]) + sum(source2[i][3])/len(source2[i][3]) + sum(source3[i][3])/len(source3[i][3]) + sum(source4[i][3])/len(source4[i][3]) + sum(source5[i][3])/len(source5[i][3])
        
    xLab.append(source1[i][1])
    yLab.append(x*0.2)
    
        
plt.figure(0)
plt.plot(xSC, ySC)
plt.title("Source Values Sports Hall")
plt.xlabel("Training Size")
plt.ylabel("CM away")   
plt.savefig("Source Values Sports Hall")  

plt.figure(1)
plt.plot(xML, yML)
plt.title("Source Values Main Lobby")
plt.xlabel("Training Size")
plt.ylabel("CM away")
plt.savefig("Source Values Main Lobby")

plt.figure(2)
plt.plot(xNC, yNC)
plt.title("Source Values Narrow Corridor")
plt.xlabel("Training Size")
plt.ylabel("CM away")  
plt.savefig("Source Values Narrow Corridor")

plt.figure(3)
plt.plot(xLab, yLab)
plt.title("Source Values Lab")
plt.xlabel("Training Size")
plt.ylabel("CM away")  
plt.savefig("Source Values Lab")


plt.figure(4)
plt.plot(xSC, ySC)
plt.plot(xML, yML)
plt.plot(xNC, yNC)
plt.plot(xLab, yLab)
plt.title("Source Values SH, ML, NC, Lab")
plt.xlabel("Training Size")
plt.ylabel("CM away")  
plt.legend(("SH", "ML", "NC", "Lab"))
plt.savefig("Source Values SH, ML, NC, Lab")