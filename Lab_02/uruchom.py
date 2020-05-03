# coding=utf8
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
import Lab02
 
# returns a compiled model
# identical to the previous one
genderModel = load_model('siec.h5')
genderModel.summary() # Display summary
 
ImgWidth = 100
ImgHeight = 100
 
ODPOWIEDZ_MEZCZYZNA = 0
ODPOWIEDZ_KOBIETA = 1
 
 
def create_image_database(dirpath):
    files = os.listdir(dirpath)
    baza_img = np.empty((len(files),ImgHeight,ImgWidth,3))
   
    i = 0
    for f in files:
        img = imageio.imread(dirpath + f)
        img = (img / 127.5) - 1
        baza_img[i,:,:,:] = img[0:ImgHeight,0:ImgWidth,0:3]
        i += 1
       
    return baza_img, files
 
 
female_database, female_files = create_image_database('./baza_testowa/K/')
male_database, male_files = create_image_database('./baza_testowa/M/')
 
female_prediction = genderModel.predict(female_database)
male_prediction = genderModel.predict(male_database)
female_prediction = [np.argmax(c) for c in keras.utils.to_categorical(female_prediction, num_classes=2)]
male_prediction = [np.argmax(c) for c in keras.utils.to_categorical(male_prediction, num_classes=2)]
 
result_str = ""
 
i = 0
result_str += "Kobiety"
result_str += "\n"
result_str += "Zdjęcie\tOdpowiedz sieci - czy kobieta?"
result_str += "\n"
for f in female_files:
    resp = "Tak" if female_prediction[i] == ODPOWIEDZ_KOBIETA else "Nie"
    result_str += f + "\t" + resp
    result_str += "\n"
    i += 1
   
result_str += "\n"
i = 0
result_str += "Mezczyzni"
result_str += "\n"
result_str += "Zdjęcie\tOdpowiedz sieci - czy mezczyzna?"
result_str += "\n"
for f in female_files:
    resp = "Tak" if male_prediction[i] == ODPOWIEDZ_MEZCZYZNA else "Nie"
    result_str += f + "\t" + resp
    result_str += "\n"
    i += 1
 
 
result_str += "\n"
 
result_str += "Dobrze sklasyfikowane kobiety: " + str(female_prediction.count(ODPOWIEDZ_KOBIETA)) + '/' + str(len(female_files))
result_str += "\n"
result_str += "Dobrze sklasyfikowani mezczyzni: " + str(male_prediction.count(ODPOWIEDZ_MEZCZYZNA)) + '/' + str(len(male_files))
result_str += "\n"
 
female_accuracy = female_prediction.count(ODPOWIEDZ_KOBIETA) / len(female_files)
male_accuracy = male_prediction.count(ODPOWIEDZ_MEZCZYZNA) / len(male_files)
overall_accuracy = (female_prediction.count(ODPOWIEDZ_KOBIETA) + male_prediction.count(ODPOWIEDZ_MEZCZYZNA)) / (len(female_files) + len(male_files))
 
result_str += "Dokladnosc klasyfikacji kobiet: " + str(female_accuracy * 100) + '%'
result_str += "\n"
result_str += "Dokladnosc klasyfikacji mezczyzn: " + str(male_accuracy * 100) + '%'
result_str += "\n"
result_str += "Ogolna dokladnosc klasyfikacji: " + str(overall_accuracy * 100) + '%'
result_str += "\n"
 
 
result_str += '''
+----------------------+------------------------+
|                      | Faktyczna płeć osoby   |
|                      | na zdjęciu             |
+----------------------+------------------------+
|                      | Kobieta    | Mężczyzna +
+----------------------+------------------------+
| Odpowiedź | Kobieta  |     {0}    |    {1}    |
+-----------+----------+------------+-----------+
| sieci     | Mężczyzna|     {2}    |    {3}    |
+----------------------+------------------------+
'''.format(female_prediction.count(ODPOWIEDZ_KOBIETA), female_prediction.count(ODPOWIEDZ_MEZCZYZNA), male_prediction.count(ODPOWIEDZ_KOBIETA), male_prediction.count(ODPOWIEDZ_MEZCZYZNA))
 
 
 
with open('tabelka.txt', 'w') as f:
    f.write(result_str)