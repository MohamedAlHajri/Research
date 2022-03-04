# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 12:09:38 2022

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt


source = np.load("CNN Large Values Code Test Run Data0, optimizer = adam, loss = mse.npy")
source1 = np.load("CNN Large Values Code Test Run Data1, optimizer = adam, loss = mse.npy")
source2 = np.load("CNN Large Values Code Test Run Data2, optimizer = adam, loss = mse.npy")
source3 = np.load("CNN Large Values Code Test Run Data3, optimizer = adam, loss = mse.npy")


    
x1env1 = []
y1env1 = []

x2env1 = []
y2env1 = []

x3env1 = []
y3env1 = []

x4env1 = []
y4env1 = []

x5env1 = []
y5env1 = []

x6env1 = []
y6env1 = []

#############

x1env2 = []
y1env2 = []

x2env2 = []
y2env2 = []

x3env2 = []
y3env2 = []

x4env2 = []
y4env2 = []

x5env2 = []
y5env2 = []

x6env2 = []
y6env2 = []

##############

x1env3 = []
y1env3 = []

x2env3 = []
y2env3 = []

x3env3 = []
y3env3 = []

x4env3 = []
y4env3 = []

x5env3 = []
y5env3 = []

x6env3 = []
y6env3 = []

##################
x1env4 = []
y1env4 = []

x2env4 = []
y2env4 = []

x3env4 = []
y3env4 = []

x4env4 = []
y4env4 = []

x5env4 = []
y5env4 = []

x6env4 = []
y6env4 = []



for i in range(17):
    x1env1.append(source[i][0][0])
    y1env1.append((source[i][0][1] + source1[i][0][1] + source2[i][0][1] + source3[i][0][1])*0.25)
    
    x2env1.append(source[i][1][0])
    y2env1.append((source[i][1][1] + source1[i][1][1] + source2[i][1][1] + source3[i][1][1])*0.25)
    
    x3env1.append(source[i][2][0])
    y3env1.append((source[i][2][1] + source1[i][2][1] + source2[i][2][1] + source3[i][2][1])*0.25)
    
    x4env1.append(source[i][3][0])
    y4env1.append((source[i][3][1] + source1[i][3][1] + source2[i][3][1] + source3[i][3][1])*0.25)
    
    x5env1.append(source[i][4][0])
    y5env1.append((source[i][4][1] + source1[i][4][1] + source2[i][4][1] + source3[i][4][1])*0.25)
    
    x6env1.append(source[i][5][0])
    y6env1.append((source[i][5][1] + source1[i][5][1] + source2[i][5][1] + source3[i][5][1])*0.25)

for i in range(17,34):
    x1env2.append(source[i][0][0])
    y1env2.append((source[i][0][1] + source1[i][0][1] + source2[i][0][1] + source3[i][0][1])*0.25)

    x2env2.append(source[i][1][0])
    y2env2.append((source[i][1][1] + source1[i][1][1] + source2[i][1][1] + source3[i][1][1])*0.25)
    
    x3env2.append(source[i][2][0])
    y3env2.append((source[i][2][1] + source1[i][2][1] + source2[i][2][1] + source3[i][2][1])*0.25)
    
    x4env2.append(source[i][3][0])
    y4env2.append((source[i][3][1] + source1[i][3][1] + source2[i][3][1] + source3[i][3][1])*0.25)
    
    x5env2.append(source[i][4][0])
    y5env2.append((source[i][4][1] + source1[i][4][1] + source2[i][4][1] + source3[i][4][1])*0.25)
    
    x6env2.append(source[i][5][0])
    y6env2.append((source[i][5][1] + source1[i][5][1] + source2[i][5][1] + source3[i][5][1])*0.25)
    
for i in range(34,51):
    x1env3.append(source[i][0][0])
    y1env3.append((source[i][0][1] + source1[i][0][1] + source2[i][0][1] + source3[i][0][1])*0.25)

    x2env3.append(source[i][1][0])
    y2env3.append((source[i][1][1] + source1[i][1][1] + source2[i][1][1] + source3[i][1][1])*0.25)
    
    x3env3.append(source[i][2][0])
    y3env3.append((source[i][2][1] + source1[i][2][1] + source2[i][2][1] + source3[i][2][1])*0.25)
    
    x4env3.append(source[i][3][0])
    y4env3.append((source[i][3][1] + source1[i][3][1] + source2[i][3][1] + source3[i][3][1])*0.25)
    
    x5env3.append(source[i][4][0])
    y5env3.append((source[i][4][1] + source1[i][4][1] + source2[i][4][1] + source3[i][4][1])*0.25)
    
    x6env3.append(source[i][5][0])
    y6env3.append((source[i][5][1] + source1[i][5][1] + source2[i][5][1] + source3[i][5][1])*0.25)
       
for i in range(51,68):
    x1env4.append(source[i][0][0])
    y1env4.append((source[i][0][1] + source1[i][0][1] + source2[i][0][1] + source3[i][0][1])*0.25)

    x2env4.append(source[i][1][0])
    y2env4.append((source[i][1][1] + source1[i][1][1] + source2[i][1][1] + source3[i][1][1])*0.25)
    
    x3env4.append(source[i][2][0])
    y3env4.append((source[i][2][1] + source1[i][2][1] + source2[i][2][1] + source3[i][2][1])*0.25)
    
    x4env4.append(source[i][3][0])
    y4env4.append((source[i][3][1] + source1[i][3][1] + source2[i][3][1] + source3[i][3][1])*0.25)
    
    x5env4.append(source[i][4][0])
    y5env4.append((source[i][4][1] + source1[i][4][1] + source2[i][4][1] + source3[i][4][1])*0.25)
    
    x6env4.append(source[i][5][0])
    y6env4.append((source[i][5][1] + source1[i][5][1] + source2[i][5][1] + source3[i][5][1])*0.25)

plt.figure(1)
plt.plot(x2env1, y2env1)
plt.title("Source = ML")
plt.xlabel("Target Size")
plt.ylabel("RMSE")
plt.legend(("Lab Mean", "NC Mean", "SH Mean"))
plt.savefig("Enviroment2 Target Values mean with ML as source Figure")       