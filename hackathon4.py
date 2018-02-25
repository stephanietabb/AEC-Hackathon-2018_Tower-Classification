# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.models import load_model, save_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import glob, os

def classify_dir(my_model_name, my_dir):
	
	model = load_model(my_model_name)
	
	print ('.'*100)
	print (' '*10)
	print (' '*10)
	print ('Asian, europian, Modern')
	for filename in glob.glob(my_dir+'/*.jpg'):
		test_image = image.load_img(filename, target_size = (64, 64))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)*1./255.
		result = model.predict_proba(test_image)
		print (filename, " ============> ", result[0][0], result[0][1], result[0][2])
	print (' '*100)
	print (' '*100)
	print ('.'*100)
	print (' '*100)


	test_datagen = ImageDataGenerator(rescale = 1./255)
	testing_set = test_datagen.flow_from_directory('testing_set', target_size = (64, 64), batch_size = 1, shuffle=False)
	filenames = testing_set.filenames 
	nb_samples = len(filenames)
	print(filenames)
	predict = model.predict_generator(testing_set, steps = nb_samples)
	print(predict)

model_name = 'h10.h5'
try: 
	classifier = load_model(model_name)	
except:
	classifier = Sequential()
	classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	classifier.add(Conv2D(32, (2, 2), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	classifier.add(Flatten())
	classifier.add(Dense(units = 16, activation = 'relu'))
	classifier.add(Dense(3, activation='softmax'))
	classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	
	train_datagen = ImageDataGenerator(
		rescale = 1./255, 
		shear_range = 0.2, 
		zoom_range = 0.2, 
		horizontal_flip = False,
		rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2)
	training_set = train_datagen.flow_from_directory('training_set', target_size = (64, 64), batch_size = 32, class_mode = 'categorical', save_to_dir='g')
	testing_datagen = ImageDataGenerator(rescale = 1./255)
	testing_set = testing_datagen.flow_from_directory('testing_set', target_size = (64, 64), batch_size = 1, class_mode = 'categorical')
	
	classifier.fit_generator(training_set, steps_per_epoch = 300, epochs = 5, validation_data = testing_set)
	save_model(classifier, model_name)

classify_dir(model_name, '/Users/fhedayat/Desktop/rnn/ht/validation_set')