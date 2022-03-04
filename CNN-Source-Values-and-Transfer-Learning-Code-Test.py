# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 00:14:18 2022

@author: User
"""

from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
import keras
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import math
Data = []

for env in range(4):
    if (env == 0):
        alldata = np.load("Complex_S21_Sport_Hall.npy")
        alldata1 = np.load("Complex_S21_Lab_139.npy");
        env1 = 'Lab 139'
        alldata2 = np.load("Complex_S21_Narrow_Corridor_71.npy");
        env2 = 'NC'
        alldata3 = np.load("Complex_S21_Main_Lobby_71.npy");
        env3 = 'ML'
        
    elif (env == 1):
        alldata = np.load("Complex_S21_Main_Lobby_71.npy")
        alldata1 = np.load("Complex_S21_Lab_139.npy");
        env1 = 'Lab 139'
        alldata2 = np.load("Complex_S21_Narrow_Corridor_71.npy");
        env2 = 'NC'
        alldata3 = np.load("Complex_S21_Sport_Hall.npy");
        env3 = 'SC'
        
    elif (env == 2):
        alldata = np.load("Complex_S21_Narrow_Corridor_71.npy")
        alldata1 = np.load("Complex_S21_Lab_139.npy");
        env1 = 'Lab 139'
        alldata2 = np.load("Complex_S21_Main_Lobby_71.npy");
        env2 = 'ML'
        alldata3 = np.load("Complex_S21_Sport_Hall.npy");
        env3 = 'SC'
        
    elif (env == 3):
        alldata = np.load("Complex_S21_Lab_139.npy")
        alldata1 = np.load("Complex_S21_Narrow_Corridor_71.npy");
        env1 = 'NC'
        alldata2 = np.load("Complex_S21_Main_Lobby_71.npy");
        env2 = 'ML'
        alldata3 = np.load("Complex_S21_Sport_Hall.npy");
        env3 = 'SC'


    alldata = np.transpose(alldata, (0, 2, 1))
    alldata = np.reshape(alldata, (601, 1960))
    alldata = np.transpose(alldata, (1, 0))

    alldata1 = np.transpose(alldata1, (0, 2, 1))
    alldata1 = np.reshape(alldata1, (601, 1960))
    alldata1 = np.transpose(alldata1, (1, 0))

    alldata2 = np.transpose(alldata2, (0, 2, 1))
    alldata2 = np.reshape(alldata2, (601, 1960))
    alldata2 = np.transpose(alldata2, (1, 0))

    alldata3 = np.transpose(alldata3, (0, 2, 1))
    alldata3 = np.reshape(alldata3, (601, 1960))
    alldata3 = np.transpose(alldata3, (1, 0))

    all_x = np.stack([np.real(alldata), np.imag(alldata)], axis=-1)
    all_x1 = np.stack([np.real(alldata1), np.imag(alldata1)], axis=-1)
    all_x2 = np.stack([np.real(alldata2), np.imag(alldata2)], axis=-1)
    all_x3 = np.stack([np.real(alldata3), np.imag(alldata3)], axis=-1)

    all_y = []
    all_y1 = []
    all_y2 = []
    all_y3 = []

    for i in range(1960):
        consider10repet = i // 10
        all_y.append([consider10repet % 14, consider10repet // 14])  # grid is 14 by 14

    all_y = np.array(all_y)
    all_y1 = np.array(all_y)
    all_y2 = np.array(all_y)
    all_y3 = np.array(all_y)
    
    def step_decay(epoch):
        if epoch > 80:
            lrate = 0.0001
        else:
            lrate = 0.005
        return lrate

    def step_decay1(epoch):
        if epoch > 80:
            lrate = 0.0001
        else:
            lrate = 0.005
        return lrate

    def step_decay2(epoch):
        if epoch > 80:
            lrate = 0.00001
        else:
            lrate = 0.0005
        return lrate


    results_all = []
    testTarget_all = []
    trainTime = []
    testTime = []

    for train_size in range(1):
        if (train_size == 0):
            train_portion = 1470  # 75%
            itera_max = 4
            itera_max1 = 17
            start1 = 0


        for target_size in range(start1, itera_max1):
            RMSE1 = []
            RMSE2 = []
            RMSE3 = []
            RMSE4 = []

            if (target_size == 0):
                target_portion = 20  #~1%
                
            elif (target_size == 1):
                target_portion = 49  #~2.5%
                
            elif (target_size == 2):
                target_portion = 98  #5%
                
            elif (target_size == 3):
                target_portion = 196  #10%
                
            elif (target_size == 4):
                target_portion = 294  #15%
                
            elif (target_size == 5):
                target_portion = 392  #20%
                
            elif (target_size == 6):
                target_portion = 490  #25%
                
            elif (target_size == 7):
                target_portion = 588  #30%
                
            elif (target_size == 8):
                target_portion = 686  #35%
                
            elif (target_size == 9):
                target_portion = 784  #40%                
                
            elif (target_size == 10):
                target_portion = 882  #45%                
                
            elif (target_size == 11):
                target_portion = 980  #50%

            elif (target_size == 12):
                target_portion = 1078  #55%       
                       
            elif (target_size == 13):
                target_portion = 1176  #60%                       
                
            elif (target_size == 14):
                target_portion = 1274  #65%                       
                
            elif (target_size == 15):
                target_portion = 1372  #70%  

            elif (target_size == 16):
                target_portion = 1470  #75%  

                
            for iTimes in range(4):  # 10 iterations
                # divide data for training and testing
                print('Iteration: ', iTimes)
                allInd = np.random.choice(1960, 1960, replace=False)
                train_Ind = allInd[:train_portion]
                test_Ind = allInd[train_portion:]

                allInd1 = np.random.choice(1960, 1960, replace=False)
                train_Ind11 = allInd1[:train_portion]  # train_portion
                test_Ind11 = allInd1[train_portion:]

                train_Ind1 = allInd[:target_portion]
                unsup_Ind1 = allInd[target_portion:1470]
                test_Ind1 = allInd[1470:]

                # This is the train & test data for the same environment
                trainData = all_x[train_Ind, :, :]
                testData = all_x[test_Ind, :, :]
                trainTarget = all_y[train_Ind, :]
                testTarget = all_y[test_Ind, :]

                # This is the train & test data for the different environments
                #          trainData1_unsupervisied = all_x1[train_Ind,:,:] #source label data location
                trainData1_unsupervisied = all_x1[unsup_Ind1, :, :]  # no location permutated
                #          trainData1_unsupervisied = all_x1[train_Ind1,:,:] #Target label data location
                trainData1 = all_x1[train_Ind1, :, :]
                testData1 = all_x1[test_Ind1, :, :]
                trainTarget1 = all_y1[train_Ind1, :]
                testTarget1 = all_y1[test_Ind1, :]

                #          trainData2_unsupervisied = all_x2[train_Ind,:,:] #source label data location
                trainData2_unsupervisied = all_x2[unsup_Ind1, :, :]  # no location permutated
                #          trainData2_unsupervisied = all_x2[train_Ind1,:,:] #Target label data location
                trainData2 = all_x2[train_Ind1, :, :]
                testData2 = all_x2[test_Ind1, :, :]
                trainTarget2 = all_y2[train_Ind1, :]
                testTarget2 = all_y2[test_Ind1, :]

                #          trainData3_unsupervisied = all_x3[train_Ind,:,:] #source label data location
                trainData3_unsupervisied = all_x3[unsup_Ind1, :, :]  # no location permutated
                #          trainData3_unsupervisied = all_x3[train_Ind1,:,:] #Target label data location
                trainData3 = all_x3[train_Ind1, :, :]
                testData3 = all_x3[test_Ind1, :, :]
                trainTarget3 = all_y3[train_Ind1, :]
                testTarget3 = all_y3[test_Ind1, :]

                trainData = np.reshape(trainData, (trainData.shape[0], trainData.shape[1], trainData.shape[2], 1))
                testData = np.reshape(testData, (testData.shape[0], testData.shape[1], testData.shape[2], 1))

                trainData1 = np.reshape(trainData1, (trainData1.shape[0], trainData1.shape[1], trainData1.shape[2], 1))
                trainData1_unsupervisied = np.reshape(trainData1_unsupervisied, (
                trainData1_unsupervisied.shape[0], trainData1_unsupervisied.shape[1], trainData1_unsupervisied.shape[2],
                1))
                testData1 = np.reshape(testData1, (testData1.shape[0], testData1.shape[1], testData1.shape[2], 1))

                trainData2 = np.reshape(trainData2, (trainData2.shape[0], trainData2.shape[1], trainData2.shape[2], 1))
                trainData2_unsupervisied = np.reshape(trainData2_unsupervisied, (
                trainData2_unsupervisied.shape[0], trainData2_unsupervisied.shape[1], trainData2_unsupervisied.shape[2],
                1))
                testData2 = np.reshape(testData2, (testData2.shape[0], testData2.shape[1], testData2.shape[2], 1))

                trainData3 = np.reshape(trainData3, (trainData3.shape[0], trainData3.shape[1], trainData3.shape[2], 1))
                trainData3_unsupervisied = np.reshape(trainData3_unsupervisied, (
                trainData3_unsupervisied.shape[0], trainData3_unsupervisied.shape[1], trainData3_unsupervisied.shape[2],
                1))
                testData3 = np.reshape(testData3, (testData3.shape[0], testData3.shape[1], testData3.shape[2], 1))

                model_input1 = Input(shape=(trainData.shape[1], trainData.shape[2], 1), name='input_11')
                transformer = Conv2D(1, kernel_size=(601, 1), strides=(1, 1), padding='same', activation='relu',
                                     name='conv2d_transformer')(model_input1)
                cnn1 = Conv2D(16, kernel_size=(10, 1), strides=(1, 1), activation='relu', name='conv2d_1')(
                    transformer)  # model_input
                pool1 = MaxPooling2D(pool_size=(3, 1), name='max_pooling2d_1')(cnn1)
                cnn2 = Conv2D(32, (10, 1), activation='relu', name='conv2d_2')(pool1)
                pool2 = MaxPooling2D(pool_size=(3, 1), padding='same', name='max_pooling2d_2')(cnn2)
                flat = Flatten(name='flatten_1')(pool2)
                dense_1 = Dense(100, activation='relu', name='dense_1')(flat)
                drop1 = Dropout(0.5, name='dropout_1')(dense_1)
                model_outputs = Dense(2, activation='linear', name='dense_2')(drop1)
                model = Model(model_input1, model_outputs)

                model.compile(optimizer='adam', loss='mse', metrics=['mse'])

                batch_size = 32  # 64
                nb_epoch = 100  # 400 200

                lrate = LearningRateScheduler(step_decay)
                callbacklist = [lrate]

                history = model.fit(trainData, trainTarget, batch_size=batch_size, verbose=0, epochs=nb_epoch,
                                    callbacks=callbacklist)

                myPred = model.predict(testData)

                myPreddiff = myPred - testTarget
                MSE1 = np.square(myPreddiff).mean()
                RMSE1.append(math.sqrt(MSE1))
                model.save("my_model")

                reconstructed_model2 = keras.models.load_model("my_model")

                reconstructed_model2.layers[0].trainable = True
                reconstructed_model2.layers[1].trainable = True
                reconstructed_model2.layers[2].trainable = True
                reconstructed_model2.layers[3].trainable = True
                reconstructed_model2.layers[4].trainable = True
                reconstructed_model2.layers[5].trainable = True
                reconstructed_model2.layers[6].trainable = True
                reconstructed_model2.layers[7].trainable = True
                reconstructed_model2.layers[8].trainable = True
                reconstructed_model2.layers[9].trainable = True

                reconstructed_model2.compile(optimizer='adam', loss='mse', metrics=['mse'])
                #          print(reconstructed_model2.summary())
                lrate1 = LearningRateScheduler(step_decay1)
                callbacklist1 = [lrate1]
                history = reconstructed_model2.fit(trainData1, trainTarget1, batch_size=batch_size, verbose=0,
                                                   epochs=nb_epoch, callbacks=callbacklist1)

                myPred = reconstructed_model2.predict(testData1)
                myPreddiff = myPred - testTarget1
                MSE2 = np.square(myPreddiff).mean()
                RMSE2.append(math.sqrt(MSE2))

                reconstructed_model2.save("reconstructed_model2")

                ###################################################
                reconstructed_model5 = keras.models.load_model("my_model")
                reconstructed_model5.layers[0].trainable = True
                reconstructed_model5.layers[1].trainable = True
                reconstructed_model5.layers[2].trainable = True
                reconstructed_model5.layers[3].trainable = True
                reconstructed_model5.layers[4].trainable = True
                reconstructed_model5.layers[5].trainable = True
                reconstructed_model5.layers[6].trainable = True
                reconstructed_model5.layers[7].trainable = True
                reconstructed_model5.layers[8].trainable = True
                reconstructed_model5.layers[9].trainable = True

                reconstructed_model5.compile(optimizer='adam', loss='mse', metrics=['mse'])
                # print(reconstructed_model1.summary())
                lrate1 = LearningRateScheduler(step_decay1)
                callbacklist1 = [lrate1]
                history = reconstructed_model5.fit(trainData2, trainTarget2, batch_size=batch_size, verbose=0,
                                                   epochs=nb_epoch, callbacks=callbacklist1)

                myPred = reconstructed_model5.predict(testData2)
                myPreddiff = myPred - testTarget2
                MSE3 = np.square(myPreddiff).mean()
                RMSE3.append(math.sqrt(MSE3))

                reconstructed_model5.save("reconstructed_model5")

                ######################################
                # We will use the unlabelled data to try to do some embedding matching or output matching
                reconstructed_model8 = keras.models.load_model("my_model")
                reconstructed_model8.layers[0].trainable = True
                reconstructed_model8.layers[1].trainable = True
                reconstructed_model8.layers[2].trainable = True
                reconstructed_model8.layers[3].trainable = True
                reconstructed_model8.layers[4].trainable = True
                reconstructed_model8.layers[5].trainable = True
                reconstructed_model8.layers[6].trainable = True
                reconstructed_model8.layers[7].trainable = True
                reconstructed_model8.layers[8].trainable = True
                reconstructed_model8.layers[9].trainable = True

                reconstructed_model8.compile(optimizer='adam', loss='mse', metrics=['mse'])
                # print(reconstructed_model8.summary())
                lrate1 = LearningRateScheduler(step_decay1)
                callbacklist1 = [lrate1]
                history = reconstructed_model8.fit(trainData3, trainTarget3, batch_size=batch_size, verbose=0,
                                                   epochs=nb_epoch, callbacks=callbacklist1)

                myPred = reconstructed_model8.predict(testData3)
                myPreddiff = myPred - testTarget3
                MSE4 = np.square(myPreddiff).mean()
                RMSE4.append(math.sqrt(MSE4))
                reconstructed_model8.save("reconstructated_model8")

                reconstructed_model6 = keras.models.load_model("reconstructated_model8")

                K.clear_session()

            Trainentry = (env, train_portion, np.mean(RMSE1))
            env1Entry = (env1, target_portion, np.mean(RMSE2))
            env2Entry = (env2, target_portion, np.mean(RMSE3))
            env3Entry = (env3, target_portion, np.mean(RMSE4))

            Data.append((Trainentry, env1Entry, env2Entry, env3Entry))
            print(Data)
            
np.save("CNN Source Values and Transfer Learning Code Test Run Data3, optimizer = adam, loss = mse", Data)
