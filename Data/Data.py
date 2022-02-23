# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:16:59 2022

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# #Source Values Plot
# 
# source = np.load("CNN Source Values Code Test Run Data0, optimizer = adam, loss = mse.npy")
# 
# x_valuesenv1 = []
# y_valuesenv1 = []
#
# 
# x_valuesenv2 = []
# y_valuesenv2 = []
# 
# x_valuesenv3 = []
# y_valuesenv3 = []
# 
# x_valuesenv4 = []
# y_valuesenv4 = []
# 
# for i in source[0:15]:
#     x_valuesenv1.append(i[1])
#     y_valuesenv1.append(i[2])
# 
# for i in source[15:30]:
#     x_valuesenv2.append(i[1])
#     y_valuesenv2.append(i[2])
# 
# for i in source[30:45]:
#     x_valuesenv3.append(i[1])
#     y_valuesenv3.append(i[2])
# 
# for i in source[45:]:
#     x_valuesenv4.append(i[1])
#     y_valuesenv4.append(i[2])
# 
# plt.figure(0)
# plt.scatter(x_valuesenv1, y_valuesenv1)
# plt.xlabel("Train Size")
# plt.ylabel("RMSE")
# plt.legend(("Sport Hall",))
# plt.savefig("Sport Hall Source Values Figure")
# 
# plt.figure(1)
# plt.scatter(x_valuesenv2, y_valuesenv2)
# plt.xlabel("Train Size")
# plt.ylabel("RMSE")
# plt.legend(("Main Lobby",))
# plt.savefig("Main Lobby Source Values Figure")
# 
# 
# plt.figure(2)
# plt.scatter(x_valuesenv3, y_valuesenv3)
# plt.xlabel("Train Size")
# plt.ylabel("RMSE")
# plt.legend(("Narror Corridor",))
# plt.savefig("Narrow Corridor Source Values Figure")
# 
# 
# plt.figure(3)
# plt.scatter(x_valuesenv4, y_valuesenv4)
# plt.xlabel("Train Size")
# plt.ylabel("RMSE")
# plt.legend(("Lab",))
# plt.savefig("Lab Source Values Figure")
# =============================================================================

# =============================================================================
# #Transfer Learning Plot
# 
# transfer = np.load("CNN Source Values and Transfer Learning Code Test Run0 Data, optimizer = adam, loss = mse.npy")
# 
# x_valuesenv1 = []
# y_Valuesenv1Source = []
# y_valuesenv1transfer1 = []
# y_valuesenv1transfer2 = []
# y_valuesenv1transfer3 = []
# 
# x_valuesenv2 = []
# y_Valuesenv2Source = []
# y_valuesenv2transfer1 = []
# y_valuesenv2transfer2 = []
# y_valuesenv2transfer3 = []
# 
# x_valuesenv3 = []
# y_Valuesenv3Source = []
# y_valuesenv3transfer1 = []
# y_valuesenv3transfer2 = []
# y_valuesenv3transfer3 = []
# 
# x_valuesenv4 = []
# y_Valuesenv4Source = []
# y_valuesenv4transfer1 = []
# y_valuesenv4transfer2 = []
# y_valuesenv4transfer3 = []
# 
# 
# for i in transfer[0:17]:
#     y_Valuesenv1Source.append(float(i[0][2]))
#     x_valuesenv1.append(float(i[1][1]))
#     y_valuesenv1transfer1.append(float(i[1][2]))
#     y_valuesenv1transfer2.append(float(i[2][2]))
#     y_valuesenv1transfer3.append(float(i[3][2]))
#     
#     
# for i in transfer[17:34]:
#     y_Valuesenv2Source.append(float(i[0][2])) 
#     x_valuesenv2.append(float(i[1][1]))
#     y_valuesenv2transfer1.append(float(i[1][2]))
#     y_valuesenv2transfer2.append(float(i[2][2]))
#     y_valuesenv2transfer3.append(float(i[3][2]))
#     
# for i in transfer[34:51]:
#     y_Valuesenv3Source.append(float(i[0][2]))
#     x_valuesenv3.append(float(i[1][1]))
#     y_valuesenv3transfer1.append(float(i[1][2]))
#     y_valuesenv3transfer2.append(float(i[2][2]))
#     y_valuesenv3transfer3.append(float(i[3][2]))
#     
# for i in transfer[51:68]:
#     y_Valuesenv4Source.append(float(i[0][2]))
#     x_valuesenv4.append(float(i[1][1]))
#     y_valuesenv4transfer1.append(float(i[1][2]))
#     y_valuesenv4transfer2.append(float(i[2][2]))
#     y_valuesenv4transfer3.append(float(i[3][2]))
#     
# #Env 1
# plt.figure(0)
# plt.scatter(x_valuesenv1, y_Valuesenv1Source)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Source Values Model - SH")
# plt.savefig("Source Values Model for Transfer Learning - SH Figure")
# 
# 
# plt.figure(1)
# plt.scatter(x_valuesenv1, y_valuesenv1transfer1)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - SH -> Lab")
# plt.savefig("Transfer Learning Model from SH to Lab Figure")
# 
# plt.figure(2)
# plt.scatter(x_valuesenv1, y_valuesenv1transfer2)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - SH -> NC")
# plt.savefig("Transfer Learning Model from SH to NC Figure")
# 
# 
# plt.figure(3)
# plt.scatter(x_valuesenv1, y_valuesenv1transfer3)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - SH -> MC")
# plt.savefig("Transfer Learning Model from SH to MC Figure")
# 
# #Env 2
# plt.figure(4)
# plt.scatter(x_valuesenv2, y_Valuesenv2Source)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Source Values Model - ML")
# plt.savefig("Source Values Model for Transfer Learning - ML Figure")
# 
# 
# plt.figure(5)
# plt.scatter(x_valuesenv2, y_valuesenv2transfer1)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - ML -> Lab")
# plt.savefig("Transfer Learning Model from ML to Lab Figure")
# 
# plt.figure(6)
# plt.scatter(x_valuesenv2, y_valuesenv2transfer2)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - ML -> NC")
# plt.savefig("Transfer Learning Model from ML to NC Figure")
# 
# 
# plt.figure(7)
# plt.scatter(x_valuesenv2, y_valuesenv2transfer3)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - ML -> SH")
# plt.savefig("Transfer Learning Model from ML to SH Figure")
# 
# #Env 3
# plt.figure(8)
# plt.scatter(x_valuesenv3, y_Valuesenv3Source)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Source Values Model - NC")
# plt.savefig("Source Values Model for Transfer Learning - NC Figure")
# 
# 
# plt.figure(9)
# plt.scatter(x_valuesenv3, y_valuesenv3transfer1)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - NC -> Lab")
# plt.savefig("Transfer Learning Model from NC to Lab Figure")
# 
# plt.figure(10)
# plt.scatter(x_valuesenv3, y_valuesenv3transfer2)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - NC -> ML")
# plt.savefig("Transfer Learning Model from NC to ML Figure")
# 
# 
# plt.figure(11)
# plt.scatter(x_valuesenv3, y_valuesenv3transfer3)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - NC -> SH")
# plt.savefig("Transfer Learning Model from NC to SH Figure")
# 
# #Env 4
# plt.figure(12)
# plt.scatter(x_valuesenv4, y_Valuesenv4Source)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Source Values Model - Lab")
# plt.savefig("Source Values Model for Transfer Learning - Lab Figure")
# 
# 
# plt.figure(13)
# plt.scatter(x_valuesenv4, y_valuesenv4transfer1)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - Lab -> NC")
# plt.savefig("Transfer Learning Model from NC to Lab Figure")
# 
# plt.figure(14)
# plt.scatter(x_valuesenv4, y_valuesenv4transfer2)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - Lab -> ML")
# plt.savefig("Transfer Learning Model from Lab to ML Figure")
# 
# 
# plt.figure(15)
# plt.scatter(x_valuesenv4, y_valuesenv4transfer3)
# plt.xlabel("Train Value")
# plt.ylabel("RMSE")
# plt.title("Transfer Learning Model - Lab -> SH")
# plt.savefig("Transfer Learning Model from Lab to SH Figure")
# =============================================================================


# =============================================================================
# #Source Values Mean Plot
# 
# source = np.load("CNN Source Values Code Test Run Data0, optimizer = adam, loss = mse.npy")
# source1 = np.load("CNN Source Values Code Test Run Data1, optimizer = adam, loss = mse.npy")
# source2 = np.load("CNN Source Values Code Test Run Data2, optimizer = adam, loss = mse.npy")
# source3 = np.load("CNN Source Values Code Test Run Data3, optimizer = adam, loss = mse.npy")
# 
# 
# x_valuesenv1 = []
# y_valuesenv1 = []
# 
# 
# x_valuesenv2 = []
# y_valuesenv2 = []
# 
# x_valuesenv3 = []
# y_valuesenv3 = []
# 
# x_valuesenv4 = []
# y_valuesenv4 = []
# 
# for i in range(15):
#     x_valuesenv1.append(source[i][1])
#     y_valuesenv1.append((source[i][2] + source1[i][2] + source2[i][2] + source3[i][2]) * 0.25)
#     print(y_valuesenv1)
# 
# for i in range(15, 30):
#     x_valuesenv2.append(source[i][1])
#     y_valuesenv2.append((source[i][2] + source1[i][2] + source2[i][2] + source3[i][2]) * 0.25)
#     
# for i in range(30, 45):
#     x_valuesenv3.append(source[i][1])
#     y_valuesenv3.append((source[i][2] + source1[i][2] + source2[i][2] + source3[i][2]) * 0.25)
#     
# for i in range(45, 60):
#     x_valuesenv4.append(source[i][1])
#     y_valuesenv4.append((source[i][2] + source1[i][2] + source2[i][2] + source3[i][2]) * 0.25)
# 
# 
# plt.figure(0)
# plt.scatter(x_valuesenv1, y_valuesenv1)
# plt.xlabel("Train Size")
# plt.ylabel("RMSE Mean")
# plt.legend(("Sport Hall Mean",))
# plt.savefig("Sport Hall Source Values Mean Figure")
# 
# plt.figure(1)
# plt.scatter(x_valuesenv2, y_valuesenv2)
# plt.xlabel("Train Size")
# plt.ylabel("RMSE Mean")
# plt.legend(("Main Lobby Mean",))
# plt.savefig("Main Lobby Source Values Mean Figure")
# 
# 
# plt.figure(2)
# plt.scatter(x_valuesenv3, y_valuesenv3)
# plt.xlabel("Train Size")
# plt.ylabel("RMSE Mean")
# plt.legend(("Narrow Corridor Mean",))
# plt.savefig("Narrow Corridor Source Values Mean Figure")
# 
# 
# plt.figure(3)
# plt.scatter(x_valuesenv4, y_valuesenv4)
# plt.xlabel("Train Size")
# plt.ylabel("RMSE")
# plt.legend(("Lab Mean",))
# plt.savefig("Lab Source Values mean Figure")
# 
# =============================================================================

source = np.load("CNN Source Values and Transfer Learning Code Test Run Data0, optimizer = adam, loss = mse.npy")
source1 = np.load("CNN Source Values and Transfer Learning Code Test Run Data1, optimizer = adam, loss = mse.npy")
source2 = np.load("CNN Source Values and Transfer Learning Code Test Run Data2, optimizer = adam, loss = mse.npy")
source3 = np.load("CNN Source Values and Transfer Learning Code Test Run Data3, optimizer = adam, loss = mse.npy")    

x_valuesenv1Source = []
y_valuesenv1Source = []

x_valuesenv1NC = []
y_valuesenv1NC = []

x_valuesenv1ML = []
y_valuesenv1ML = []

x_valuesenv1Lab = []
y_valuesenv1Lab = []


x_valuesenv2Source = []
y_valuesenv2Source = []

x_valuesenv2Lab = []
y_valuesenv2Lab = []

x_valuesenv2NC = []
y_valuesenv2NC = []

x_valuesenv2SC = []
y_valuesenv2SC = []


x_valuesenv3Source = []
y_valuesenv3Source = []

x_valuesenv3Lab = []
y_valuesenv3Lab = []

x_valuesenv3ML = []
y_valuesenv3ML = []

x_valuesenv3SC = []
y_valuesenv3SC = []


x_valuesenv4Source = []
y_valuesenv4Source = []

x_valuesenv4NC = []
y_valuesenv4NC = []

x_valuesenv4ML = []
y_valuesenv4ML = []

x_valuesenv4SC = []
y_valuesenv4SC = []


for i in range(17):
    x_valuesenv1Source.append(float(source[i][0][1]))
    y_valuesenv1Source.append((float(source[i][0][2]) + float(source1[i][0][2]) + float(source2[i][0][2]) + float(source3[i][0][2]))*0.25)
                            
    
    x_valuesenv1Lab.append(float(source[i][1][1]))
    y_valuesenv1Lab.append((float(source[i][1][2]) + float(source1[i][1][2]) + float(source2[i][1][2]) + float(source3[i][1][2]))*0.25)
    
    x_valuesenv1NC.append(float(source[i][2][1]))
    y_valuesenv1NC.append((float(source[i][2][2]) + float(source1[i][2][2]) + float(source2[i][2][2]) + float(source3[i][2][2]))*0.25)
    
    x_valuesenv1ML.append(float(source[i][3][1]))
    y_valuesenv1ML.append((float(source[i][3][2]) + float(source1[i][3][2]) + float(source2[i][3][2]) + float(source3[i][3][2]))*0.25)
    

    
for i in range(17, 34):
    x_valuesenv2Source.append(float(source[i][0][1]))
    y_valuesenv2Source.append((float(source[i][0][2]) + float(source1[i][0][2]) + float(source2[i][0][2]) + float(source3[i][0][2]))*0.25)
                            
    
    x_valuesenv2Lab.append(float(source[i][1][1]))
    y_valuesenv2Lab.append((float(source[i][1][2]) + float(source1[i][1][2]) + float(source2[i][1][2]) + float(source3[i][1][2]))*0.25)
    
    x_valuesenv2NC.append(float(source[i][2][1]))
    y_valuesenv2NC.append((float(source[i][2][2]) + float(source1[i][2][2]) + float(source2[i][2][2]) + float(source3[i][2][2]))*0.25)
    
    x_valuesenv2SC.append(float(source[i][3][1]))
    y_valuesenv2SC.append((float(source[i][3][2]) + float(source1[i][3][2]) + float(source2[i][3][2]) + float(source3[i][3][2]))*0.25)    
    
    
    
    
    
    
    
for i in range(34, 51):
    x_valuesenv3Source.append(float(source[i][0][1]))
    y_valuesenv3Source.append((float(source[i][0][2]) + float(source1[i][0][2]) + float(source2[i][0][2]) + float(source3[i][0][2]))*0.25)
                            
    
    x_valuesenv3Lab.append(float(source[i][1][1]))
    y_valuesenv3Lab.append((float(source[i][1][2]) + float(source1[i][1][2]) + float(source2[i][1][2]) + float(source3[i][1][2]))*0.25)
    
    x_valuesenv3ML.append(float(source[i][2][1]))
    y_valuesenv3ML.append((float(source[i][2][2]) + float(source1[i][2][2]) + float(source2[i][2][2]) + float(source3[i][2][2]))*0.25)
    
    x_valuesenv3SC.append(float(source[i][3][1]))
    y_valuesenv3SC.append((float(source[i][3][2]) + float(source1[i][3][2]) + float(source2[i][3][2]) + float(source3[i][3][2]))*0.25)        
    
    
    
    
    
for i in range(51, 68):
    x_valuesenv4Source.append(float(source[i][0][1]))
    y_valuesenv4Source.append((float(source[i][0][2]) + float(source1[i][0][2]) + float(source2[i][0][2]) + float(source3[i][0][2]))*0.25)
                            
    
    x_valuesenv4NC.append(float(source[i][1][1]))
    y_valuesenv4NC.append((float(source[i][1][2]) + float(source1[i][1][2]) + float(source2[i][1][2]) + float(source3[i][1][2]))*0.25)
    
    x_valuesenv4ML.append(float(source[i][2][1]))
    y_valuesenv4ML.append((float(source[i][2][2]) + float(source1[i][2][2]) + float(source2[i][2][2]) + float(source3[i][2][2]))*0.25)
    
    x_valuesenv4SC.append(float(source[i][3][1]))
    y_valuesenv4SC.append((float(source[i][3][2]) + float(source1[i][3][2]) + float(source2[i][3][2]) + float(source3[i][3][2]))*0.25)  
    
    
plt.figure(0)
plt.scatter(x_valuesenv4, y_valuesenv4)
plt.xlabel("Target Size")
plt.ylabel("RMSE")
plt.legend(("Lab Mean",))
plt.savefig("Lab Source Values mean Figure")    