import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

root_dir = 'food/'
ragu_dir = 'spaghetti_bolognese/'
carbonara_dir = 'spaghetti_carbonara/'
lasagna_dir = 'lasagna/'

all_ragu = os.listdir(os.path.join(root_dir, ragu_dir))
all_carbonara = os.listdir(os.path.join(root_dir, carbonara_dir))
#all_lasagna = os.listdir(os.path.join(root_dir, lasagna_dir))

import matplotlib.image as img
from skimage.transform import resize

target_w = 65
target_h = 65
all_imgs_orig = []
all_imgs = []
all_labels = []
idx = 0    
min_side = 400
resize_count = 0

for img_name in all_ragu:
    img_arr = img.imread(os.path.join(root_dir, ragu_dir, img_name))
    w,h,d = img_arr.shape
    img_arr_rs = img_arr
    img_arr_rs = resize(img_arr, (target_w, target_h))
    all_imgs.append(img_arr_rs)
    all_imgs_orig.append(img_arr)
    all_labels.append(1)

for img_name in all_carbonara:
    img_arr = img.imread(os.path.join(root_dir, carbonara_dir, img_name))
    w,h,d = img_arr.shape
    img_arr_rs = img_arr
    img_arr_rs = resize(img_arr, (target_w, target_h))
    all_imgs.append(img_arr_rs)
    all_imgs_orig.append(img_arr)
    all_labels.append(0)

# for img_name in all_lasagna:
#     img_arr = img.imread(os.path.join(root_dir, lasagna_dir, img_name))
#     w,h,d = img_arr.shape
#     img_arr_rs = img_arr
#     img_arr_rs = resize(img_arr, (target_w, target_h))
#     all_imgs.append(img_arr_rs)
#     all_imgs_orig.append(img_arr)
#     all_labels.append(2)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Dense, Concatenate
from keras.models import load_model, Model
from keras.optimizers import Adam, SGD, Adagrad
from keras.utils import to_categorical


from sklearn.model_selection import train_test_split


X = np.array(all_imgs)
Y = to_categorical(np.array(all_labels),num_classes=2)
Y = Y[:,0]
print(X.shape)
print(Y.shape)

from top_level_features import hog_features
from top_level_features import color_histogram_hsv
from top_level_features import extract_features
features = extract_features(X,[hog_features, color_histogram_hsv]) #extrae todo
print X.shape
print features.shape

scaler = StandardScaler()
X = scaler.fit_transform(features)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(100,input_dim=586))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(25))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
optimizer = Adam()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

batch_size = 64
n_epochs = 20

history = model.fit(X_train,Y_train,epochs=n_epochs,batch_size=batch_size,verbose=2,validation_data=(X_test, Y_test))


with open('FNN_history.history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


