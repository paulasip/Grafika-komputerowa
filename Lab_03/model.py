import plaidml.keras
plaidml.keras.install_backend()

from keras.models import Sequential
import keras
from keras.layers.core import Dense
from keras.optimizers import Adam
import csv
import numpy as np

CSV_FILENAME = 'owid-covid-data.csv'
DATE_START = '2020-03-15'
DATE_END = '2020-05-03'
INPUT_COUNT = 7
MAX_COVID_CASES = 25000

def create_network(input_count, second_layer_count, third_layer_count):
    network = Sequential()
    network.add(Dense(units=input_count, activation='relu', input_dim=input_count))
    network.add(Dense(units=second_layer_count, activation='relu'))
    network.add(Dense(units=third_layer_count, activation='relu'))
    network.add(Dense(units=1, activation=keras.activations.sigmoid))

    opt = Adam()
    network.compile(loss='mse', optimizer=opt)
    network.summary()
    
    return network


def get_data_for_country(country):
    with open(CSV_FILENAME, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data_list = []
        for row in reader:
            if row['location'] == country and row['date'] >= DATE_START and row['date'] <= DATE_END:
                data_list.append((row['location'], row['total_cases'], row['date']))
                
        return data_list
    
    
def train_network(network, data_list, epoch):
    for i in range(len(data_list)):
        if i + INPUT_COUNT >= len(data_list):
            break
                
        input_vector = []
        for j in range(i, i + INPUT_COUNT):
            input_vector.append(int(data_list[j][1]) / MAX_COVID_CASES)

        input_vector = np.asarray(input_vector).reshape(1, len(input_vector))
        target = np.asarray([int(data_list[i + INPUT_COUNT][1]) / MAX_COVID_CASES])
        network.fit(input_vector, target, epochs=epoch, verbose=1)
        
        
def train_country(network, country, epoch):
    country_data = get_data_for_country(country)
    train_network(network, country_data, epoch)
    

if __name__ == '__main__':
    network = create_network(INPUT_COUNT, 15, 15)
    
    
    train_country(network, 'Poland', 10)
    train_country(network, 'Czech Republic', 3)
    train_country(network, 'Austria', 3)
    train_country(network, 'Romania', 2)
    train_country(network, 'Denmark', 1)
    #train_country(network, 'Portugal', 2)
    #train_country(network, 'Sweden', 3)
    
    network.save('siec.h5')
        
