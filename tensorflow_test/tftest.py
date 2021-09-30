## Global variables

BATCH_SIZE = 32 # 訓練的一個Batch
EPOCHS = 50 # teacher model迭代次數(student建議100=10*EPOCHS次以上)
CATGORICAL = 10 # dataset類別總數
PATIENCE = 10 # 如果過多少個EPOCHS沒改善就停止訓練
INPUT_SHAPE = (32, 32, 3) # 圖片的size,(長,寬,幾個顏色(RGB)), 128效果太差
TARGET_SIZE = (INPUT_SHAPE[0], INPUT_SHAPE[1])  # 圖片的size,(長,寬)，這是preprosseing用的，務必保持與INPUT_SHAPE的前兩個一模一樣
TRAIN_SIZE = 0.8
VAL_SPLIT = 0.2 # validation set佔train set的比例
STUDENT_CNN_LR = 1e-5
LR_FACTOR = 0.4 # new_lr = lr * factor.
LR_PATIENCE = 3 # umber of epochs with no improvement after which learning rate will be reduced
MODEL_NAME = 'tftest.hdf5' # CNN (without KD) model name

### Setup ###

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import accuracy_score

### Load the dataset ###

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# expand new axis, channel axis 
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# [optional]: we may need 3 channel (instead of 1)
X_train = np.repeat(X_train, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

# resize the input shape , i.e. old shape: 28, new shape: 32
X_train = tf.image.resize(X_train, TARGET_SIZE) # if we want to resize 
X_test = tf.image.resize(X_test, TARGET_SIZE) # if we want to resize 

# one hot 
y_train = tf.keras.utils.to_categorical(y_train, num_classes=CATGORICAL)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=CATGORICAL)


data_augmentation_student = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(scale=1/255),
    layers.experimental.preprocessing.Normalization()
])

# images in cifar is small, use LeNet
student_CNN = keras.Sequential(
[
    data_augmentation_student,
    Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=INPUT_SHAPE, activation='tanh'),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(120, activation='tanh'),
    Dense(84, activation='tanh'),
    Dense(CATGORICAL, activation='softmax')
],
    name="student_CNN",
)
student_CNN.compile(optimizer=Adam(learning_rate=STUDENT_CNN_LR), loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath=MODEL_NAME, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='auto', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_accuracy', factor=LR_FACTOR, patience=LR_PATIENCE, 
                                   verbose=1, mode='auto', min_delta=0.0001)
early = EarlyStopping(monitor='val_accuracy', mode="auto", patience=PATIENCE)
callbacks_list = [checkpoint, reduceLROnPlat, early]
history = student_CNN.fit(X_train, y_train,
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=callbacks_list,
                    validation_split=VAL_SPLIT)

# 計算正確率
CNN_epoch = len(list(range(len(history.history['accuracy']))))
CNN_train_acc = history.history['accuracy'][-1]
CNN_val_acc = history.history['val_accuracy'][-1]
y_test_int = [np.where(r==1)[0][0] for r in y_test] # y_test is one hot, convert to integer labels
student_CNN.load_weights(MODEL_NAME)
student_prediction = student_CNN.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
y_prediction_S = np.argmax(student_prediction, axis=1)
CNN_test_acc = accuracy_score(y_prediction_S, y_test_int)


### Collect the results ###

with open('result.txt',  'a') as file_obj:
    file_obj.write('================EXP_NAME: '+ EXP_NAME+ '================' + '\n')
    file_obj.write('CNN_epoch: '         + str(CNN_epoch)         + '\n')
    file_obj.write('CNN_train_acc: '     + str(CNN_train_acc)     + '\n')
    file_obj.write('CNN_val_acc: '       + str(CNN_val_acc)       + '\n')
    file_obj.write('CNN_test_acc: '      + str(CNN_test_acc)      + '\n')