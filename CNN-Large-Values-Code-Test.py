from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
import keras
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
Data = []

def step_decay(epoch):
    if epoch > 80:
        return 0.0001
    return 0.005


def step_decay1(epoch):
    if epoch > 80:
        return 0.00001
    return 0.0005


for env in range(4):
    if (env == 0):
        alldata = np.load("Complex_S21_Sport_Hall.npy"); env0 = "SH"
        alldata1 = np.load("Complex_S21_Lab_139.npy");
        env1 = 'Lab 139'
        alldata2 = np.load("Complex_S21_Narrow_Corridor_71.npy");
        env2 = 'NC'
        alldata3 = np.load("Complex_S21_Main_Lobby_71.npy");
        env3 = 'ML'

    elif (env == 1):
        alldata = np.load("Complex_S21_Main_Lobby_71.npy"); env0 = "ML"
        alldata1 = np.load("Complex_S21_Lab_139.npy");
        env1 = 'Lab 139'
        alldata2 = np.load("Complex_S21_Narrow_Corridor_71.npy");
        env2 = 'NC'
        alldata3 = np.load("Complex_S21_Sport_Hall.npy");
        env3 = 'SC'

    elif (env == 2):
        alldata = np.load("Complex_S21_Narrow_Corridor_71.npy"); env0 = "NC"
        alldata1 = np.load("Complex_S21_Lab_139.npy");
        env1 = 'Lab 139'
        alldata2 = np.load("Complex_S21_Main_Lobby_71.npy");
        env2 = 'ML'
        alldata3 = np.load("Complex_S21_Sport_Hall.npy");
        env3 = 'SC'

    elif (env == 3):
        alldata = np.load("Complex_S21_Lab_139.npy"); env0 = "Lab"
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

    results_all = []
    testTarget_all = []
    trainTime = []
    testTime = []

    train_portion = 1470
    itera_max = 4
    itera_max1 = 17
    start1 = 0

    for target_size in range(17):
        SourceAllRMSE = []

        Target1AllNoTuningRMSE = []
        Target2AllNoTuningRMSE = []
        Target3AllNoTuningRMSE = []

        Target1AllWithTuningRMSE = []
        Target2AllWithTuningRMSE = []
        Target3AllWithTuningRMSE = []

        Target1AllWithTuningAndSemiRMSE = []
        Target2AllWithTuningAndSemiRMSE = []
        Target3AllWithTuningAndSemiRMSE = []


        testRMSE_allknn = []
        test_RMSEknn1 = []

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

        for iteration in range(4):
            print("Iteration: ", iteration)

            #Dviding the data for testing and training
            allInd = np.random.choice(1960, 1960, replace=False)
            train_Ind = allInd[:train_portion]
            test_Ind = allInd[train_portion:]

            allInd1 = np.random.choice(1960, 1960, replace=False)
            train_Ind11 = allInd1[:train_portion]  # train_portion
            test_Ind11 = allInd1[train_portion:]

            train_Ind1 = allInd[:target_portion]
            unsup_Ind1 = allInd[target_portion:1470]
            test_Ind1 = allInd[1470:]

            #This is the train & test data for the same environment
            trainData = all_x[train_Ind,:,:]
            testData = all_x[test_Ind,:,:]
            trainTarget = all_y[train_Ind,:]
            testTarget = all_y[test_Ind,:]

            # This is the train & test data for the different environments
            trainData1_unsupervisied = all_x1[unsup_Ind1, :, :]  # no location permutated
            trainData1 = all_x1[train_Ind1, :, :]
            testData1 = all_x1[test_Ind1, :, :]
            trainTarget1 = all_y1[train_Ind1, :]
            testTarget1 = all_y1[test_Ind1, :]

            trainData2_unsupervisied = all_x2[unsup_Ind1, :, :]  # no location permutated
            trainData2 = all_x2[train_Ind1, :, :]
            testData2 = all_x2[test_Ind1, :, :]
            trainTarget2 = all_y2[train_Ind1, :]
            testTarget2 = all_y2[test_Ind1, :]

            trainData3_unsupervisied = all_x3[unsup_Ind1, :, :]  # no location permutated
            trainData3 = all_x3[train_Ind1, :, :]
            testData3 = all_x3[test_Ind1, :, :]
            trainTarget3 = all_y3[train_Ind1, :]
            testTarget3 = all_y3[test_Ind1, :]

            trainData = np.reshape(trainData, (trainData.shape[0], trainData.shape[1], trainData.shape[2], 1))
            testData = np.reshape(testData, (testData.shape[0], testData.shape[1], testData.shape[2], 1))

            trainData1 = np.reshape(trainData1, (trainData1.shape[0], trainData1.shape[1], trainData1.shape[2], 1))
            trainData1_unsupervisied = np.reshape(trainData1_unsupervisied, (
            trainData1_unsupervisied.shape[0], trainData1_unsupervisied.shape[1], trainData1_unsupervisied.shape[2], 1))
            testData1 = np.reshape(testData1, (testData1.shape[0], testData1.shape[1], testData1.shape[2], 1))

            trainData2 = np.reshape(trainData2, (trainData2.shape[0], trainData2.shape[1], trainData2.shape[2], 1))
            trainData2_unsupervisied = np.reshape(trainData2_unsupervisied, (
            trainData2_unsupervisied.shape[0], trainData2_unsupervisied.shape[1], trainData2_unsupervisied.shape[2], 1))
            testData2 = np.reshape(testData2, (testData2.shape[0], testData2.shape[1], testData2.shape[2], 1))

            trainData3 = np.reshape(trainData3, (trainData3.shape[0], trainData3.shape[1], trainData3.shape[2], 1))
            trainData3_unsupervisied = np.reshape(trainData3_unsupervisied, (
            trainData3_unsupervisied.shape[0], trainData3_unsupervisied.shape[1], trainData3_unsupervisied.shape[2], 1))
            testData3 = np.reshape(testData3, (testData3.shape[0], testData3.shape[1], testData3.shape[2], 1))

            #Preparing stage of data is done
            #proceeding to model 1, source.

            #These lines are needed in case the source models need to be trained
            #In our case, we are going to use the same pre-trained source model to increase the accuracy and avoid the randomness interfering with the results
            # model_input1 = Input(shape=(trainData.shape[1], trainData.shape[2], 1), name='input_11')
            # transformer = Conv2D(1, kernel_size=(601, 1), strides=(1, 1), padding='same', activation='relu',
            #                      name='conv2d_transformer')(model_input1)
            # cnn1 = Conv2D(16, kernel_size=(10, 1), strides=(1, 1), activation='relu', name='conv2d_1')(
            #     transformer)  # model_input
            # pool1 = MaxPooling2D(pool_size=(3, 1), name='max_pooling2d_1')(cnn1)
            # cnn2 = Conv2D(32, (10, 1), activation='relu', name='conv2d_2')(pool1)
            # pool2 = MaxPooling2D(pool_size=(3, 1), padding='same', name='max_pooling2d_2')(cnn2)
            # flat = Flatten(name='flatten_1')(pool2)
            # dense_1 = Dense(100, activation='relu', name='dense_1')(flat)
            # drop1 = Dropout(0.5, name='dropout_1')(dense_1)
            # model_outputs = Dense(2, activation='linear', name='dense_2')(drop1)
            # model = Model(model_input1, model_outputs)
            #
            # model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            #
            # batch_size = 32  # 64
            # nb_epoch = 100  # 400 200
            #
            # lrate = LearningRateScheduler(step_decay)
            # callbacklist = [lrate]
            #
            # history = model.fit(trainData, trainTarget, batch_size=batch_size, verbose=0, epochs=nb_epoch,
            #                     callbacks=callbacklist)
            #

            if env0 == "SH":
                model = keras.models.load_model("SH Source Trained Model 1470")
            elif env0 == "NC":
                model = keras.models.load_model("NC Source Trained Model 1470")
            elif env0 == "ML":
                model = keras.models.load_model("ML Source Trained Model 1470")
            elif env0 == "Lab":
                model = keras.models.load_model("Lab Source Trained Model 1470")

            model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            myPred = model.predict(testData)
            myPreddiff = myPred - testTarget
            dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
            test_RMSE = 12.5*np.sqrt(np.sum(np.power(dis,2))/(len(dis))) #What does the 12.5 do

            SourceAllRMSE.append(test_RMSE)
            results_all.append(myPred)
            testTarget_all.append(testTarget)

            model.save("Model 1")


            myPred = model.predict(testData1) #For the first test target.
            myPreddiff = myPred-testTarget1
            dis = np.sqrt(np.power(myPreddiff[:,0],2)+np.power(myPreddiff[:,1],2))
            test_RMSE11 = 12.5*np.sqrt(np.sum(np.power(dis,2))/(len(dis)))
            Target1AllNoTuningRMSE.append(test_RMSE11)


            myPred = model.predict(testData2) #For the Second test target.
            myPreddiff = myPred-testTarget2
            dis = np.sqrt(np.power(myPreddiff[:,0],2)+np.power(myPreddiff[:,1],2))
            test_RMSE22 = 12.5*np.sqrt(np.sum(np.power(dis,2))/(len(dis)))
            Target2AllNoTuningRMSE.append(test_RMSE22)


            myPred = model.predict(testData3) #For the Third test target.
            myPreddiff = myPred-testTarget3
            dis = np.sqrt(np.power(myPreddiff[:,0],2)+np.power(myPreddiff[:,1],2))
            test_RMSE33 = 12.5*np.sqrt(np.sum(np.power(dis,2))/(len(dis)))
            Target3AllNoTuningRMSE.append(test_RMSE33)



            reconstructed_model1 = keras.models.load_model("Model 1") #Transfer Learning for first target
            reconstructed_model1.layers[0].trainable = True
            reconstructed_model1.layers[1].trainable = True
            reconstructed_model1.layers[2].trainable = True
            reconstructed_model1.layers[3].trainable = True
            reconstructed_model1.layers[4].trainable = True
            reconstructed_model1.layers[5].trainable = True
            reconstructed_model1.layers[6].trainable = True
            reconstructed_model1.layers[7].trainable = True
            reconstructed_model1.layers[8].trainable = True
            reconstructed_model1.layers[9].trainable = True

            reconstructed_model1.compile(optimizer = "adam", loss = "mse", metrics = ["mse"])

            lrate1 = LearningRateScheduler(step_decay)
            callbacklist1 = [lrate1]

            batch_size = 32
            nb_epoch = 100

            history = reconstructed_model1.fit(trainData1, trainTarget1, batch_size=batch_size, verbose=0, epochs=nb_epoch,
                                   callbacks=callbacklist1)


            myPred = reconstructed_model1.predict(testData1)
            myPreddiff = myPred-testTarget1
            dis = np.sqrt(np.power(myPreddiff[:,0],2)+np.power(myPreddiff[:,1],2))
            test_RMSE2 = 12.5*np.sqrt(np.sum(np.power(dis,2))/490.0)
            Target1AllWithTuningRMSE.append(test_RMSE2)

            reconstructed_model1.save("reconstructed_model1")


            reconstructed_model2 = keras.models.load_model("Model 1") #Transfer Learning for second target
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

            reconstructed_model2.compile(optimizer="adam", loss="mse", metrics=["mse"])

            lrate1 = LearningRateScheduler(step_decay)
            callbacklist1 = [lrate1]
            history = reconstructed_model2.fit(trainData1, trainTarget1, batch_size=batch_size, verbose=0,
                                               epochs=nb_epoch,
                                               callbacks=callbacklist1)

            myPred = reconstructed_model2.predict(testData2)
            myPreddiff = myPred - testTarget2
            dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
            test_RMSE2 = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / 490.0)
            Target2AllWithTuningRMSE.append(test_RMSE2)

            reconstructed_model2.save("reconstructed_model2")


            reconstructed_model3 = keras.models.load_model("Model 1")  # Transfer Learning for third target
            reconstructed_model3.layers[0].trainable = True
            reconstructed_model3.layers[1].trainable = True
            reconstructed_model3.layers[2].trainable = True
            reconstructed_model3.layers[3].trainable = True
            reconstructed_model3.layers[4].trainable = True
            reconstructed_model3.layers[5].trainable = True
            reconstructed_model3.layers[6].trainable = True
            reconstructed_model3.layers[7].trainable = True
            reconstructed_model3.layers[8].trainable = True
            reconstructed_model3.layers[9].trainable = True

            reconstructed_model3.compile(optimizer="adam", loss="mse", metrics=["mse"])

            lrate1 = LearningRateScheduler(step_decay)
            callbacklist1 = [lrate1]
            history = reconstructed_model3.fit(trainData1, trainTarget1, batch_size=batch_size, verbose=0,
                                               epochs=nb_epoch,
                                               callbacks=callbacklist1)

            myPred = reconstructed_model3.predict(testData3)
            myPreddiff = myPred - testTarget3
            dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
            test_RMSE2 = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / 490.0)
            Target3AllWithTuningRMSE.append(test_RMSE2)

            reconstructed_model3.save("reconstructed_model3")

            """
            What should be added is the repeated change in the size of the traindata.
            Add a for loop make it something like 1470 * 0.05n maybe
            """


            ##Psuedolabel and refinement for first target environment
            if target_portion < 1470:


                reconstructed_model4 = keras.models.load_model("Model 1")
                reconstructed_model5 = keras.models.load_model("reconstructed_model1")

                trainData1_unsupervisied1 = trainData1_unsupervisied
                trainData1_unsupervisied1 = np.reshape(trainData1_unsupervisied1,[trainData1_unsupervisied1.shape[0],trainData1_unsupervisied1.shape[1],trainData1_unsupervisied1.shape[2],1])

                reconstructed_model4.layers[0].trainable = True
                reconstructed_model4.layers[1].trainable = True
                reconstructed_model4.layers[2].trainable = True
                reconstructed_model4.layers[3].trainable = True
                reconstructed_model4.layers[4].trainable = True
                reconstructed_model4.layers[5].trainable = True
                reconstructed_model4.layers[6].trainable = True
                reconstructed_model4.layers[7].trainable = True
                reconstructed_model4.layers[8].trainable = True
                reconstructed_model4.layers[9].trainable = True
                reconstructed_model4.compile(optimizer='adam', loss= 'mse', metrics=['mse'])


                model_output11 = reconstructed_model5.get_layer("dense_2").output
                m11 = Model(inputs=reconstructed_model5.input, outputs=model_output11)
                proxy_label_data11 = m11.predict(trainData1)
                cur_dim11 = proxy_label_data11.shape[1]


                model_input2 = Input(shape=(cur_dim11,), name='input_11')
                Input1 = Dense(40, activation='relu', name='input_layer')(model_input2)
                Intermediate1 = Dense(40, activation='relu', name='intermediate_layer')(Input1)
                Final_output1 = Dense(2, activation='linear', name='output1')(Intermediate1)
                refinment_model = Model(model_input2, Final_output1)
                refinment_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
                lrate1 = LearningRateScheduler(step_decay)
                callbacklist1 = [lrate1]
                refinment_model.fit(proxy_label_data11, trainTarget1, batch_size=batch_size, verbose=0, epochs=nb_epoch,
                                    callbacks=callbacklist1)

                refinment_model.save('refinement')


                refinemnet_model1 = keras.models.load_model("refinement")


                model_output11 = reconstructed_model5.get_layer("dense_2").output
                m11 = Model(inputs=reconstructed_model5.input, outputs=model_output11)
                pseudo_label_data12 = m11.predict(trainData1_unsupervisied1)

                model_output12 = refinemnet_model1.get_layer("output1").output
                m12 = Model(inputs=refinemnet_model1.input, outputs=model_output12)
                pseudo_label_data_refined1 = m12.predict(pseudo_label_data12) #Shouldnt it be traindata1_unsupervised and then compare refined vs not refined?

                all_data1 = np.concatenate((trainData1, trainData1_unsupervisied1))
                all_data_label1 = np.concatenate((trainTarget1,pseudo_label_data_refined1))

                lrate1 = LearningRateScheduler(step_decay1)
                callbacklist1 = [lrate1]
                history = reconstructed_model4.fit(all_data1, all_data_label1, batch_size=batch_size, verbose=0, epochs=nb_epoch,callbacks=callbacklist1)

                reconstructed_model4.save("reconstructated4_model")

                myPred = reconstructed_model4.predict(testData1)
                myPreddiff = myPred-testTarget1
                dis = np.sqrt(np.power(myPreddiff[:,0],2)+np.power(myPreddiff[:,1],2))
                Target1AllWithTuningAndSemiRMSE = 12.5*np.sqrt(np.sum(np.power(dis,2))/490.0)

                ##Psuedolabel and refinement for second target environment

                reconstructed_model4 = keras.models.load_model("Model 1")
                reconstructed_model5 = keras.models.load_model("reconstructed_model1")

                trainData2_unsupervisied1 = trainData2_unsupervisied
                trainData2_unsupervisied1 = np.reshape(trainData2_unsupervisied1, [trainData2_unsupervisied1.shape[0],
                                                                                   trainData2_unsupervisied1.shape[1],
                                                                                   trainData2_unsupervisied1.shape[2], 1])

                reconstructed_model4.layers[0].trainable = True
                reconstructed_model4.layers[1].trainable = True
                reconstructed_model4.layers[2].trainable = True
                reconstructed_model4.layers[3].trainable = True
                reconstructed_model4.layers[4].trainable = True
                reconstructed_model4.layers[5].trainable = True
                reconstructed_model4.layers[6].trainable = True
                reconstructed_model4.layers[7].trainable = True
                reconstructed_model4.layers[8].trainable = True
                reconstructed_model4.layers[9].trainable = True
                reconstructed_model4.compile(optimizer='adam', loss='mse', metrics=['mse'])

                model_output21 = reconstructed_model5.get_layer("dense_2").output
                m21 = Model(inputs=reconstructed_model5.input, outputs=model_output21)
                proxy_label_data21 = m21.predict(trainData2)
                cur_dim21 = proxy_label_data21.shape[1]

                model_input2 = Input(shape=(cur_dim21,), name='input_11')
                Input1 = Dense(40, activation='relu', name='input_layer')(model_input2)
                Intermediate1 = Dense(40, activation='relu', name='intermediate_layer')(Input1)
                Final_output1 = Dense(2, activation='linear', name='output1')(Intermediate1)
                refinment_model = Model(model_input2, Final_output1)
                refinment_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
                lrate1 = LearningRateScheduler(step_decay)
                callbacklist1 = [lrate1]
                refinment_model.fit(proxy_label_data21, trainTarget2, batch_size=batch_size, verbose=0, epochs=nb_epoch,
                                    callbacks=callbacklist1)

                refinment_model.save('refinement')

                refinemnet_model1 = keras.models.load_model("refinement")

                model_output21 = reconstructed_model5.get_layer("dense_2").output
                m21 = Model(inputs=reconstructed_model5.input, outputs=model_output21)
                pseudo_label_data22 = m21.predict(trainData2_unsupervisied1)

                model_output22 = refinemnet_model1.get_layer("output1").output
                m22 = Model(inputs=refinemnet_model1.input, outputs=model_output22)
                pseudo_label_data_refined21 = m22.predict(
                    pseudo_label_data22)  # Shouldnt it be traindata1_unsupervised and then compare refined vs not refined?

                all_data2 = np.concatenate((trainData2, trainData2_unsupervisied1))
                all_data_label2 = np.concatenate((trainTarget2, pseudo_label_data_refined21))

                lrate1 = LearningRateScheduler(step_decay1)
                callbacklist1 = [lrate1]
                history = reconstructed_model4.fit(all_data2, all_data_label2, batch_size=batch_size, verbose=0,
                                                   epochs=nb_epoch, callbacks=callbacklist1)

                reconstructed_model4.save("reconstructated4_model")

                myPred = reconstructed_model4.predict(testData2)
                myPreddiff = myPred - testTarget2
                dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
                Target2AllWithTuningAndSemiRMSE = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / 490.0)


                ##Psuedolabel and refinement for third target environment


                reconstructed_model4 = keras.models.load_model("Model 1")
                reconstructed_model5 = keras.models.load_model("reconstructed_model1")

                trainData3_unsupervisied1 = trainData3_unsupervisied
                trainData3_unsupervisied1 = np.reshape(trainData3_unsupervisied1, [trainData3_unsupervisied1.shape[0],
                                                                                   trainData3_unsupervisied1.shape[1],
                                                                                   trainData3_unsupervisied1.shape[2], 1])

                reconstructed_model4.layers[0].trainable = True
                reconstructed_model4.layers[1].trainable = True
                reconstructed_model4.layers[2].trainable = True
                reconstructed_model4.layers[3].trainable = True
                reconstructed_model4.layers[4].trainable = True
                reconstructed_model4.layers[5].trainable = True
                reconstructed_model4.layers[6].trainable = True
                reconstructed_model4.layers[7].trainable = True
                reconstructed_model4.layers[8].trainable = True
                reconstructed_model4.layers[9].trainable = True
                reconstructed_model4.compile(optimizer='adam', loss='mse', metrics=['mse'])

                model_output31 = reconstructed_model5.get_layer("dense_2").output
                m31 = Model(inputs=reconstructed_model5.input, outputs=model_output31)
                proxy_label_data31 = m31.predict(trainData3)
                cur_dim31 = proxy_label_data31.shape[1]

                model_input2 = Input(shape=(cur_dim31,), name='input_11')
                Input1 = Dense(40, activation='relu', name='input_layer')(model_input2)
                Intermediate1 = Dense(40, activation='relu', name='intermediate_layer')(Input1)
                Final_output1 = Dense(2, activation='linear', name='output1')(Intermediate1)
                refinment_model = Model(model_input2, Final_output1)
                refinment_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
                lrate1 = LearningRateScheduler(step_decay)
                callbacklist1 = [lrate1]
                refinment_model.fit(proxy_label_data31, trainTarget3, batch_size=batch_size, verbose=0, epochs=nb_epoch,
                                    callbacks=callbacklist1)

                refinment_model.save('refinement')

                refinemnet_model1 = keras.models.load_model("refinement")

                model_output31 = reconstructed_model5.get_layer("dense_2").output
                m31 = Model(inputs=reconstructed_model5.input, outputs=model_output31)
                pseudo_label_data32 = m31.predict(trainData3_unsupervisied1)

                model_output32 = refinemnet_model1.get_layer("output1").output
                m32 = Model(inputs=refinemnet_model1.input, outputs=model_output32)
                pseudo_label_data_refined3 = m32.predict(
                    pseudo_label_data32)  # Shouldnt it be traindata1_unsupervised and then compare refined vs not refined?

                all_data3 = np.concatenate((trainData3, trainData3_unsupervisied1))
                all_data_label3 = np.concatenate((trainTarget3, pseudo_label_data_refined3))

                lrate1 = LearningRateScheduler(step_decay1)
                callbacklist1 = [lrate1]
                history = reconstructed_model4.fit(all_data3, all_data_label3, batch_size=batch_size, verbose=0,
                                                   epochs=nb_epoch, callbacks=callbacklist1)

                reconstructed_model4.save("reconstructated4_model")

                myPred = reconstructed_model4.predict(testData2)
                myPreddiff = myPred - testTarget3
                dis = np.sqrt(np.power(myPreddiff[:, 0], 2) + np.power(myPreddiff[:, 1], 2))
                Target3AllWithTuningAndSemiRMSE = 12.5 * np.sqrt(np.sum(np.power(dis, 2)) / 490.0)

        Data.append((target_portion, SourceAllRMSE, Target1AllNoTuningRMSE, Target2AllNoTuningRMSE, Target3AllNoTuningRMSE, Target1AllWithTuningRMSE, Target2AllWithTuningRMSE, Target3AllWithTuningRMSE,
                     Target1AllWithTuningAndSemiRMSE, Target2AllWithTuningAndSemiRMSE, Target3AllWithTuningAndSemiRMSE))
        print(Data)
np.save("Large Values Run", Data)