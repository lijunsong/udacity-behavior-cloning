import time
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Activation, MaxPooling2D, Dropout, Cropping2D

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline


### define utility for read data and preprocess data
def read_image(path):
    i = cv2.imread(path.strip())
    return cv2.cvtColor(i, cv2.COLOR_BGR2RGB)

def show_image(img):
    plt.imshow(img)

def explore_data(data):
    plt.hist(data['steering'])

def read_csv(csv_file):
    return pd.read_csv(csv_file, header=None,
                       names=['center', 'left', 'right', 'steering', 'shrottle', 'brake', 'speed'])

def get_training_file_path(csv_file, frac):
    """read in a csv file, frac indicates what fraction of data that has steering 0 will be used."""
    content = read_csv(csv_file)
    center = content[['center', 'steering']]
    # add image and steering from left and right camera.
    left = content[['left', 'steering']]
    right = content[['right', 'steering']]
    # add angle +/- 0.15 to steering angel
    left['steering'] = left['steering'].map(lambda x: x+0.15 if x+0.15 <= 1.0 else 1.0)
    assert(not (left['steering']==content['steering']).all())
    right['steering'] = right['steering'].map(lambda x: x-0.15 if x-0.15>=-1.0 else -1.0)
    assert(not (right['steering']==content['steering']).all())
    left.columns = ['center', 'steering']
    right.columns = ['center', 'steering']

    data = pd.concat([left, center, right])
    assert(data.shape[1] == 2)

    # filter out steering angle that is 0
    zero_steering = data[data['steering'].map(lambda x: -0.001<=x<=0.001)]
    zero_sample = zero_steering.sample(frac=frac)
    nonzero_steering = data[data['steering'].map(lambda x: x>0.001 or x<-0.001)]
    path_in_use = nonzero_steering.append(zero_sample)
    return path_in_use[['center', 'steering']]

def more_paths_from_csv(csv, frac=0.3, other_paths=None):
    data_paths = get_training_file_path(csv, frac)
    if other_paths is not None:
        all_paths = [other_paths, data_paths]
    else:
        all_paths = [data_paths]
    return pd.concat(all_paths)

def readimg_and_preprocess(paths):
    imgs = np.array([read_image(p) for p in paths])
    return imgs

# split the data to train and test
def get_data_generator(data, batch_size=64):
    def generate_data(training_data):
        "given dataFrame from csv file, return a (X_train,y_train) generator"
        N = len(training_data)
        paths = training_data['center'].values
        steering  = training_data['steering'].values
        while 1:
            paths, steering = shuffle(paths, steering)
            for offset in range(0, N, batch_size):
                # need to convert array of object to 4-D array
                X_train = np.array([read_image(p) for p in paths[offset:offset+batch_size]])
                y_train = steering[offset:offset+batch_size]
                yield (X_train, y_train)
                # flip the data
                X_train = np.array([np.fliplr(i) for i in X_train])
                y_train = y_train * (-1.0)
                yield (X_train, y_train)
    "return two generators on training set and validation set"
    train_data, validation_data = train_test_split(data)
    train_gen = generate_data(train_data)
    validation_gen = generate_data(validation_data)
    # double the size because we've flipped the data
    train_size = len(train_data) * 2
    validation_size = len(validation_data) * 2
    return (train_gen, train_size), (validation_gen, validation_size)


# parameters
input_shape = (160, 320, 3)
filters = 6
batch_size = 128

def model_init(fromfile=None):
    "return a model"
    if fromfile:
        model = load_model(fromfile)
        return model

    model = Sequential()
    # crop top 75 and bottom 25, last channel at index 3
    model.add(Cropping2D(cropping=((75,25),(0,0)), input_shape=(160,320,3)))
    # normalize
    model.add(Lambda(lambda x: x/255.0-0.5))

    model.add(Convolution2D(24, 5, 5, subsample=(2,2), input_shape=(None,60,320,3), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 3, 3, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def model_fit(model, training_data, batch_size=64, modelfile=None):
    ((train_gen, train_gen_size), (validation_gen, validation_gen_size)) = \
        get_data_generator(training_data, batch_size)
    model.fit_generator(train_gen, samples_per_epoch=train_gen_size,
			nb_epoch=4,
			validation_data=validation_gen,
			nb_val_samples=validation_gen_size)
    if not modelfile:
        modelfile="model-%d.h5"%int(time.time())
    model.save(modelfile)
    print(modelfile + " saved.")
    return model

# to use
# model = model_init('model.h5')
# new_data = more_data_from_csv('.../driving_log.csv', frac=...)
# new_model = model_fit(model, new_data, batch_size=..., modelfile=...)

if __name__ == '__main__':
    # start from beginning
    model = model_init()
    paths = more_paths_from_csv('data/driving_log.csv', frac=0.8)
    #     add more paths here
    paths = more_paths_from_csv('recover/driving_log.csv', other_paths=paths)
    #     visualize using
    #explore_data(paths)
    #     fit
    new_model = model_fit(model, paths, batch_size=128, modelfile='model.h5')
