from keras.layers import  Dense, Dropout,  Input, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import math

Data = []
for env in range(4):
  if(env==0):
    alldata = np.load("Complex_S21_Sport_Hall.npy")

  elif(env==1):
    alldata = np.load("Complex_S21_Main_Lobby_71.npy")

  elif(env==2):
    alldata = np.load("Complex_S21_Narrow_Corridor_71.npy")

  if(env==3):
    alldata = np.load("Complex_S21_Lab_139.npy")

  alldata = np.transpose(alldata,(0,2,1))
  alldata = np.reshape(alldata,(601,1960))
  alldata = np.transpose(alldata,(1,0))
  all_x = np.stack([np.real(alldata), np.imag(alldata)], axis=-1)


  all_y = []


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

  results_all = []
  testTarget_all = []
  trainTime = []
  testTime = []

  itera_max = 4

  for train_size in range(15):
    if(train_size==0):
      train_portion = 1470 #75%

    elif(train_size==0):
      train_portion = 1372 #70%

    elif(train_size==1):
      train_portion = 1274 #65%

    elif(train_size==2):
      train_portion = 1176 #60%

    elif(train_size==3):
      train_portion = 1078 #55%

    elif(train_size==4):
      train_portion = 980 #50%

    if(train_size==5) :
      train_portion = 882 #45%

    elif(train_size==6):
      train_portion = 784 #40%

    elif(train_size==7):
      train_portion = 686 #35%

    elif(train_size==8):
      train_portion = 588 #30%

    elif(train_size==9):
      train_portion = 490 #25%

    elif(train_size==10):
      train_portion = 392 #20%

    elif(train_size==11):
      train_portion = 294 #15%

    elif(train_size==12):
      train_portion = 196 #10%

    elif(train_size==13):
      train_portion = 98 #5%

    elif(train_size==14):
      train_portion = 49 #~2.5%

    elif(train_size==15):
      train_portion = 20 #~1%

    RMSE = []

    for iTimes in range(itera_max):   # 10 iterations
        #divide data for training and testing
        print('Iteration: ', iTimes)
        allInd = np.random.choice(1960, 1960, replace=False)
        train_Ind = allInd[:train_portion]
        test_Ind = allInd[train_portion:]

        allInd1 = np.random.choice(1960, 1960, replace=False)
        train_Ind11 = allInd1[:train_portion] #train_portion
        test_Ind11 = allInd1[train_portion:]

        #This is the train & test data for the same environment
        trainData = all_x[train_Ind,:,:]
        testData = all_x[test_Ind,:,:]
        trainTarget = all_y[train_Ind,:]
        testTarget = all_y[test_Ind,:]


        trainData = np.reshape(trainData,(trainData.shape[0],trainData.shape[1],trainData.shape[2],1))
        testData = np.reshape(testData,(testData.shape[0],testData.shape[1],testData.shape[2],1))

        model_input1 = Input(shape=(trainData.shape[1], trainData.shape[2],1),name = 'input_11')
        transformer = Conv2D(1, kernel_size=(601, 1), strides=(1, 1),padding='same',activation='relu',name = 'conv2d_transformer')(model_input1)
        cnn1 = Conv2D(16, kernel_size=(10, 1), strides=(1, 1),activation='relu',name = 'conv2d_1')(transformer) #    model_input
        pool1 = MaxPooling2D(pool_size=(3, 1),name = 'max_pooling2d_1')(cnn1)
        cnn2 = Conv2D(32, (10, 1), activation='relu',name = 'conv2d_2')(pool1)
        pool2 = MaxPooling2D(pool_size=(3, 1),padding='same',name = 'max_pooling2d_2')(cnn2)
        flat = Flatten(name = 'flatten_1')(pool2)
        dense_1 = Dense(100, activation='relu',name = 'dense_1')(flat)
        drop1 = Dropout(0.5,name = 'dropout_1')(dense_1)
        model_outputs = Dense(2, activation='linear',name = 'dense_2')(drop1)
        model = Model(model_input1, model_outputs)

        model.compile(optimizer='adam',loss= 'mse', metrics=['mse'])

        batch_size =  32 #64
        nb_epoch = 100 #400 200

        lrate = LearningRateScheduler(step_decay)
        callbacklist = [lrate]

        history = model.fit(trainData, trainTarget, batch_size=batch_size, verbose=0, epochs=nb_epoch,callbacks=callbacklist)

        myPred = model.predict(testData)
        myPreddiff = myPred - testTarget
        MSE1 = np.square(myPreddiff).mean()
        RMSE.append(math.sqrt(MSE1))

        model.save("CNN Source values test")

        K.clear_session()

    print('Mean test results:',np.mean(np.array(RMSE)),'Environment',env,'train size:',train_portion)
    Data.append((env, train_portion, np.mean(np.array(RMSE))))

np.save("CNN Source Values Code Test Run Data3, optimizer = adam, loss = mse", Data)
