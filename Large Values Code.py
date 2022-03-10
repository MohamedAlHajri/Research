# -*- coding: utf-8 -*-
"""
@author: Khalifa Alsuwaidi
"""

from keras.layers import  Dense, Dropout,  Input, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
import keras
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from scipy.spatial import distance_matrix
import math
Data = []
for env in range(4):
    if(env==0):
        alldata = np.load("Complex_S21_Sport_Hall.npy"); env0 = 'SC'
        alldata1 = np.load("Complex_S21_Lab_139.npy"); env1 = 'Lab 139'
        alldata2 = np.load("Complex_S21_Narrow_Corridor_71.npy"); env2 = 'NC'
        alldata3 = np.load("Complex_S21_Main_Lobby_71.npy"); env3 = 'ML'
    
    elif(env==1):
        alldata = np.load("Complex_S21_Main_Lobby_71.npy"); env0 = 'ML
        alldata1 = np.load("Complex_S21_Lab_139.npy"); env1 = 'Lab 139'
        alldata2 = np.load("Complex_S21_Narrow_Corridor_71.npy"); env2 = 'NC'
        alldata3 = np.load("Complex_S21_Sport_Hall.npy"); env3 = 'SC'   
    
    elif(env==2):
        alldata = np.load("Complex_S21_Narrow_Corridor_71.npy"); env0 = 'NC'
        alldata1 = np.load("Complex_S21_Lab_139.npy"); env1 = 'Lab 139'
        alldata2 = np.load("Complex_S21_Main_Lobby_71.npy") ; env2 = 'ML'
        alldata3 = np.load("Complex_S21_Sport_Hall.npy") ; env3 = 'SC'  
    
    elif(env==3):
        alldata = np.load("Complex_S21_Lab_139.npy"); env0 = 'Lab 139'
        alldata1 = np.load("Complex_S21_Narrow_Corridor_71.npy"); env1 = 'NC'
        alldata2 = np.load("Complex_S21_Main_Lobby_71.npy"); env2 = 'ML'
        alldata3 = np.load("Complex_S21_Sport_Hall.npy") ; env3 = 'SC'
    
    alldata = np.transpose(alldata,(0,2,1))
    alldata = np.reshape(alldata,(601,1960))
    alldata = np.transpose(alldata,(1,0))
  
    alldata1 = np.transpose(alldata1,(0,2,1))
    alldata1 = np.reshape(alldata1,(601,1960))
    alldata1 = np.transpose(alldata1,(1,0))

    alldata2 = np.transpose(alldata2,(0,2,1))
    alldata2 = np.reshape(alldata2,(601,1960))
    alldata2 = np.transpose(alldata2,(1,0))

    alldata3 = np.transpose(alldata3,(0,2,1))
    alldata3 = np.reshape(alldata3,(601,1960))
    alldata3 = np.transpose(alldata3,(1,0))
  
    all_x = np.stack([np.real(alldata), np.imag(alldata)], axis=-1)
    all_x1 = np.stack([np.real(alldata1), np.imag(alldata1)], axis=-1)
    all_x2 = np.stack([np.real(alldata2), np.imag(alldata2)], axis=-1)
    all_x3 = np.stack([np.real(alldata3), np.imag(alldata3)], axis=-1)
  
    all_y = []
    all_y1 = []
    all_y2 = []
    all_y3 = []
  
    for i in range(1960):
        consider10repet = i//10
        all_y.append([consider10repet%14,consider10repet//14]) # grid is 14 by 14
  
    all_y = np.array(all_y)
    all_y1 = np.array(all_y)
    all_y2 = np.array(all_y)
    all_y3 = np.array(all_y)
  
    def step_decay(epoch):
        if epoch>80:
            lrate = 0.0001
        else:
            lrate = 0.005
        return lrate
  
    def step_decay1(epoch):
        if epoch>80:
            lrate = 0.0001
        else:
            lrate = 0.005
        return lrate
    
    def step_decay2(epoch):
      if epoch>80:
          lrate = 0.00001
      else:
          lrate = 0.0005
      return lrate
        
    results_all = []
    testTarget_all = []
    trainTime = []
    testTime = []

    train_portion = 1470  # 75%
    itera_max = 4
    itera_max1 = 17
    start1 = 0

    for target_size in range(start1, itera_max1):
        testRMSE_all = []
        testRMSE_all1 = []
        testRMSE_all2 = []
        testRMSE_all3 = []
        testRMSE_all4 = []
        testRMSE_all5 = []
        testRMSE_all6 = []
        testRMSE_all7 = []
        testRMSE_all8 = []
        testRMSE_all11 = []
        testRMSE_all22 = []
        testRMSE_all33 = []
        testRMSE_allknn = []
        test_RMSEknn1 = []
        RMSE1 = []
        RMSE2 = []
        RMSE3 = []
        RMSE4 = []
        RMSE5 = []
        RMSE6 = []

        if (target_size == 0):
            target_portion = 20  # ~1%

        elif (target_size == 1):
            target_portion = 49  # ~2.5%

        elif (target_size == 2):
            target_portion = 98  # 5%

        elif (target_size == 3):
            target_portion = 196  # 10%

        elif (target_size == 4):
            target_portion = 294  # 15%

        elif (target_size == 5):
            target_portion = 392  # 20%

        elif (target_size == 6):
            target_portion = 490  # 25%

        elif (target_size == 7):
            target_portion = 588  # 30%

        elif (target_size == 8):
            target_portion = 686  # 35%

        elif (target_size == 9):
            target_portion = 784  # 40%

        elif (target_size == 10):
            target_portion = 882  # 45%

        elif (target_size == 11):
            target_portion = 980  # 50%

        elif (target_size == 12):
            target_portion = 1078  # 55%

        elif (target_size == 13):
            target_portion = 1176  # 60%

        elif (target_size == 14):
            target_portion = 1274  # 65%

        elif (target_size == 15):
            target_portion = 1372  # 70%

        elif (target_size == 16):
            target_portion = 1470  # 75%


        for iTimes in range(itera_max):  # 10 iterations
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

            if env0 == 'MC': #First Stage
                SourceModel = keras.models.load_model("1470 Source Model MC")
            elif env0 == 'NC':
                SourceModel = keras.models.load_model("1470 Source Model NC")
            elif env0 == 'SC':
                SourceModel = keras.models.load_model("1470 Source Model SC")
            elif env0 == 'Lab 139':
                SourceModel = keras.models.load_model("1470 Source Model Lab 139")


            SourceModel.layers[0].trainable = True
            SourceModel.layers[1].trainable = True
            SourceModel.layers[2].trainable = True
            SourceModel.layers[3].trainable = True
            SourceModel.layers[4].trainable = True
            SourceModel.layers[5].trainable = True
            SourceModel.layers[6].trainable = True
            SourceModel.layers[7].trainable = True
            SourceModel.layers[8].trainable = True
            SourceModel.layers[9].trainable = True
            
            reconstructed_model2.compile(optimizer='adam', loss='mse', metrics=['mse'])
            lrate1 = LearningRateScheduler(step_decay1)
            callbacklist1 = [lrate1]
            
            
            batch_size = 32  # 64
            nb_epoch = 100  # 400 200

            lrate = LearningRateScheduler(step_decay)
            callbacklist = [lrate]

            
            myPred = SourceModel.predict(testData) #Testing test data0 and calcualting RMSE from source
            myPreddiff = myPred - testTarget
            dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
            test_RMSE = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / (len(dis)))
            testRMSE_all.append(test_RMSE)
            results_all.append(myPred)
            testTarget_all.append(testTarget)

            MSE1 = np.square(myPreddiff).mean()
            RMSE1.append(math.sqrt(MSE1))
            SourceModel.save("my_model")

            myPred = SourceModel.predict(testData1) #testing data1 and calcualting RMSE from target1
            myPreddiff = myPred - testTarget1
            dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
            test_RMSE11 = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / (len(dis)))
            testRMSE_all11.append(test_RMSE11)
            
            MSE2 = np.square(myPreddiff).mean()
            RMSE2.append(math.sqrt(MSE2))



            myPred = SourceModel.predict(testData2) #testing data2 and calculating RMSE from target2
            myPreddiff = myPred - testTarget2
            dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
            test_RMSE22 = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / (len(dis)))
            testRMSE_all22.append(test_RMSE22)

            MSE3 = np.square(myPreddiff).mean()
            RMSE3.append(math.sqrt(MSE3))


            myPred = SourceModel.predict(testData3) #testing data3 and calculating RMSE from target3
            myPreddiff = myPred - testTarget3
            dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
            test_RMSE33 = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / (len(dis)))
            testRMSE_all33.append(test_RMSE33)
            
            MSE4 = np.square(myPreddiff).mean()
            RMSE4.append(math.sqrt(MSE4))


####################################################### Second Stage
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
            lrate1 = LearningRateScheduler(step_decay1)
            callbacklist1 = [lrate1]
            history = reconstructed_model2.fit(trainData1, trainTarget1, batch_size=batch_size, verbose=0,
                                               epochs=nb_epoch, callbacks=callbacklist1)

            myPred = reconstructed_model2.predict(testData1) #test data 1 only
            myPreddiff = myPred - testTarget1
            dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
            test_RMSE2 = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / 490.0)
            testRMSE_all2.append(test_RMSE2)
            
            MSE5 = np.square(myPreddiff).mean()
            RMSE5.append(math.sqrt(MSE5))

            reconstructed_model2.save("reconstructed_model2")

######################### Third Stage
            k12 = 1
            error_trace = 0
            reconstructed_model = keras.models.load_model("my_model")
            reconstructed_model1 = keras.models.load_model("reconstructed_model2")

            trainData1_unsupervisied1 = trainData1_unsupervisied
            trainData1_unsupervisied1 = np.reshape(trainData1_unsupervisied1,
                                                   [trainData1_unsupervisied1.shape[0],
                                                    trainData1_unsupervisied1.shape[1],
                                                    trainData1_unsupervisied1.shape[2], 1])


            reconstructed_model.layers[0].trainable = True
            reconstructed_model.layers[1].trainable = True
            reconstructed_model.layers[2].trainable = True
            reconstructed_model.layers[3].trainable = True
            reconstructed_model.layers[4].trainable = True
            reconstructed_model.layers[5].trainable = True
            reconstructed_model.layers[6].trainable = True
            reconstructed_model.layers[7].trainable = True
            reconstructed_model.layers[8].trainable = True
            reconstructed_model.layers[9].trainable = True

            reconstructed_model.compile(optimizer='adam', loss='mse', metrics=['mse'])

            # proxy label stage (maybe we need to increase expressive power)
            model_output = reconstructed_model1.get_layer("dense_2").output
            m = Model(inputs=reconstructed_model1.input, outputs=model_output)
            proxy_label_data = m.predict(trainData1)
            cur_dim = proxy_label_data.shape[1]
            # proxy_label_data = np.reshape(proxy_label_data,(proxy_label_data.shape[1],proxy_label_data.shape[0],1))


            model_input2 = Input(shape=(cur_dim,), name='input_11')
            Input1 = Dense(40, activation='relu', name='input_layer')(model_input2)
            Intermediate1 = Dense(40, activation='relu', name='intermediate_layer')(Input1)
            Final_output1 = Dense(2, activation='linear', name='output1')(Intermediate1)
            refinment_model = Model(model_input2, Final_output1)
            refinment_model.compile(optimizer='adam', loss='mse', metrics=['mse'])

            lrate1 = LearningRateScheduler(step_decay1)
            callbacklist1 = [lrate1]

            refinment_model.fit(proxy_label_data, trainTarget1, batch_size=batch_size, verbose=0,
                                 epochs=nb_epoch, callbacks=callbacklist1) 

            refinment_model.save('refinemnet')
             # pseudo label stage
            refinemnet_model1 = keras.models.load_model("refinemnet")
            
            

            model_output = reconstructed_model1.get_layer("dense_2").output
            m = Model(inputs=reconstructed_model1.input, outputs=model_output)
            pseudo_label_data = m.predict(trainData1_unsupervisied1) #not labeled

            model_output1 = refinemnet_model1.get_layer("output1").output
            m1 = Model(inputs=refinemnet_model1.input, outputs=model_output1)
            pseudo_label_data_refined = m1.predict(pseudo_label_data) #labeled

            all_data = np.concatenate((trainData1, trainData1_unsupervisied1))
            all_data_label = np.concatenate((trainTarget1, pseudo_label_data_refined)) #generate the psuedocode


            lrate1 = LearningRateScheduler(step_decay1)
            callbacklist1 = [lrate1]
            history = reconstructed_model.fit(all_data, all_data_label, batch_size=batch_size, verbose=0,
                                              epochs=nb_epoch, callbacks=callbacklist1)

            reconstructed_model.save("reconstructated_model")

            myPred = reconstructed_model.predict(testData1)
            myPreddiff = myPred - testTarget1
            dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
            test_RMSE1 = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / 490.0)
            
            MSE6 = np.square(myPreddiff).mean()
            RMSE6.append(math.sqrt(MSE6))

            error_trace = test_RMSE1
            testRMSE_all1.append(error_trace)

            K.clear_session()



        env11 = (train_portion, np.mean(RMSE1))
        env22 = (target_portion, np.mean(RMSE2))
        env33 = (target_portion, np.mean(RMSE3))
        env44 = (target_portion, np.mean(RMSE4))
        env55 = (target_portion, np.mean(RMSE5))
        env66 = (target_portion, np.mean(RMSE6))

        Data.append((env11, env22, env33, env44, env55, env66))
        print(Data)
        
np.save("CNN Large Values Code Test Run Data1, optimizer = adam, loss = mse", Data)
