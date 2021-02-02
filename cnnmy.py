#Convolutional Neural Network
#Installing Keras
#Building CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#Initializing CNN
classifier=Sequential()
#Creating Covolutional Network
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Adding second convolutional layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Flattening
classifier.add(Flatten())
#Full Connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
#Compiling
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Fitting CNN into images

from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory('dataset/training_set',
                                               target_size=(64,64),
                                               batch_size=32,
                                               class_mode='binary')
test_set=test_datagen.flow_from_directory('dataset/test_set',
                                          target_size=(64,64),
                                          batch_size=32,
                                          class_mode='binary')
classifier.fit_generator(training_set,
                        samples_per_epoch=8000,
                        nb_epoch=25,
                        validation_data=test_set,
                        nb_val_samples=2000)
#Predicting a new image
 import numpy as np                       
from keras.preprocessing import image as image_utils
test_image = image_utils.load_img('dataset/single_prediction/cat_or_dog.jpg',
target_size = (64, 64))
test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict_on_batch(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'                        
                        
 prediction                       
                        
                        


