import plaidml.keras
plaidml.keras.install_backend()

import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import load_model
import model

# do 3 maja
historical_data = [11617, # 26 kwietnia 
                   11902, 
                   12218, 
                   12640, 
                   12877, 
                   13105, 
                   13375, # 3 maja
                   ]


for i in range(len(historical_data)):
    historical_data[i] /= model.MAX_COVID_CASES
    

network = load_model('siec.h5')

print("Przewidywania:")
for i in range(15):
    prediction = network.predict(np.asarray(historical_data[i:i + model.INPUT_COUNT]).reshape(1, model.INPUT_COUNT))
    prediction = prediction[0][0]
    historical_data.append(prediction)
    print(str(3 + i + 1) + ' maja. Przypadki: ' + str(int(prediction * model.MAX_COVID_CASES)))


