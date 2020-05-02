#Sieć neuronowa

#Bibioteki do obliczen tensorowych

#import tensorflow as tf
#from tensorflow import keras

import plaidml.keras
plaidml.keras.install_backend()

#Bibioteka do obsługi sieci neuronowych
import keras

#Załadowania bazy uczącej
import imageio
import numpy as np

import os

from keras.utils.np_utils import to_categorical
from keras.models import load_model

# returns a compiled model
# identical to the previous one
genderModel = load_model('siec.h5')
genderModel.summary() # Display summary

ImgWidth = 100
ImgHeight = 100

BazaImg = np.empty((10,ImgHeight,ImgWidth,3))

FileName = ".\\baza_testowa\\K\\21.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[0,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\22.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[1,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\23.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[2,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\24.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[3,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\K\\25.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[4,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

#------------------------------------------------------------

FileName = ".\\baza_testowa\\M\\21.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[5,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\22.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[6,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\23.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[7,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\24.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[8,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

FileName = ".\\baza_testowa\\M\\25.jpg"
Img = imageio.imread(FileName)
Img = (Img / 127.5) - 1
BazaImg[9,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]

gender = genderModel.predict(BazaImg) # 0 - m, 1 - k
print([np.argmax(c) for c in keras.utils.to_categorical(gender, num_classes=2)])
print(gender)


#50
#--------------------------------
# Prawdziwa Płeć | Odpowiedz sieci
#----------------+---------+------
#                |   M     |  K
#----------------+---------+------
#   M            |         |  1
#----------------+---------+------
#   K            |         |  1

#ImgCount = 100
#ImgWidth = 100
#ImgHeight = 100
#
#BazaImg = np.empty((ImgCount,ImgHeight,ImgWidth,3))
#BazaAns = np.empty((ImgCount)) # 0 - m, 1 - k
#dirList = os.listdir(".\\baza\\m")
#i=0
#for dir in dirList:
#  print(dir)
#  FileName = ".\\baza\\m\\{}".format(dir)
#  Img = imageio.imread(FileName)
#  Img = (Img / 127.5) - 1
#  BazaImg[i,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]
#  BazaAns[i] = 0
#  i=i+1
#  if i>=ImgCount/2:
#    break
#
#dirList = os.listdir(".\\baza\\k")
#for dir in dirList:
#  print(dir)
#  FileName = ".\\baza\\k\\{}".format(dir)
#  Img = imageio.imread(FileName)
#  Img = (Img / 127.5) - 1
#  BazaImg[i,:,:,:] = Img[0:ImgHeight,0:ImgWidth,0:3]
#  BazaAns[i] = 1
#  i=i+1
#  if i>=ImgCount:
#    break
#
#BazaImg = BazaImg[0:i,:,:,:]
#BazaAns = BazaAns[0:i]
#print(BazaImg.shape)
#
##Stworzenia modelu sieci
#
#input  = keras.engine.input_layer.Input(shape=(ImgHeight,ImgWidth,3),name="wejscie")
#
#FlattenLayer = keras.layers.Flatten()
#
#path = FlattenLayer(input)
#
#for i in range(0,6):
#  LayerDense1 = keras.layers.Dense(50, activation=None, use_bias=True, kernel_initializer='glorot_uniform')
#  path = LayerDense1(path)
#
#  LayerPReLU1 = keras.layers.PReLU(alpha_initializer='zeros', shared_axes=None)
#  path = LayerPReLU1(path)
#
#LayerDenseN = keras.layers.Dense(1, activation=keras.activations.sigmoid, use_bias=True, kernel_initializer='glorot_uniform')
#output = LayerDenseN(path)
#
##---------------------------------
## Creation of TensorFlow Model
##---------------------------------
#genderModel = keras.Model(input, output, name='genderEstimatior')
#
#genderModel.summary() # Display summary
#
##Włączenia procesu uczenia
#
#rmsOptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#
#genderModel.compile(optimizer=rmsOptimizer,loss=keras.losses.binary_crossentropy,metrics=['accuracy'])
#
#genderModel.fit(BazaImg, BazaAns, epochs=15, batch_size=10, shuffle=True)
#
#genderModel.save('siec.h5')
##Przetestować / użyć sieci
#
