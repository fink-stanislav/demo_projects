
import numpy as np

from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dense

import keras.applications.xception as xc
import keras.applications.resnet50 as r50

import cv2

class Classifier(object):

    def __init__(self):
        self.root_dir = '/home/stas/dev/demo_projects/dog_breed_classifier'
        
        with open(self._get_dir('resources/dog_names.txt'), 'r') as f:
            lines = f.readlines()
        self.dog_names = [line.strip() for line in lines]
 
        self.model = self.create_transferred_model()
        self.face_cascade = cv2.CascadeClassifier(self._get_dir('resources/haarcascade_frontalface_alt.xml'))
        self.ResNet50_model = r50.ResNet50(weights='imagenet')

    def _get_dir(self, path):
        return '{}/{}'.format(self.root_dir, path)

    def face_detector(self, img):
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def create_transferred_model(self):
        transferred_model = Sequential()
        transferred_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        transferred_model.add(Dense(133, activation='softmax'))
        path = self._get_dir('resources/weights.best.Xception.hdf5')
        transferred_model.load_weights(path)
        return transferred_model

    def extract_Xception(self, tensor):
        return xc.Xception(weights='imagenet', include_top=False).predict(xc.preprocess_input(tensor))

    def img_to_tensor(self, img):
        # loads RGB image as PIL.Image.Image type
        #img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        img = img.resize((224, 224), Image.ANTIALIAS)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def predict_dog_breed(self, model, img, dog_names):
        tensor = self.img_to_tensor(img)
        bottleneck_feature = self.extract_Xception(tensor)
        # obtain predicted vector
        predicted_vector = model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]

    def ResNet50_predict_labels(self, img):
        # returns prediction vector for image located at img_path
        img = r50.preprocess_input(self.img_to_tensor(img))
        return np.argmax(self.ResNet50_model.predict(img))

    def dog_detector(self, img):
        prediction = self.ResNet50_predict_labels(img)
        return ((prediction <= 268) & (prediction >= 151)) 

    def detect_creature(self, img):
        if (self.dog_detector(img)):
            return 'dog'
        elif self.face_detector(img):
            return 'human'
        return 'neither'

    def make_prediction(self, img):
        detected = self.detect_creature(img)
        result = {}
        result[detected] = self.predict_dog_breed(self.model, img, self.dog_names)
        return result

    def prettify_breed(self, breed):
        if '_' in breed:
            splits = breed.split('_')
            return ' '.join(splits)
        else:
            return breed

    def interact(self, img):
        prediction = self.make_prediction(img)
    
        keys = ['dog', 'human', 'neither']    
        for key in keys:
            breed = prediction.pop(key, None)
            if breed is not None:
                breed = self.prettify_breed(breed)
                if key == 'human':
                    return ('This is a human that resembles {}'.format(breed))
                elif key == 'dog':
                    return ('This is a dog, possibly {}'.format(breed))
                else:
                    return ('This thing is unknown for me but it resembles {}'.format(breed))

classifier = Classifier()
