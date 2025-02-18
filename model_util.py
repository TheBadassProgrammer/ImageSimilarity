import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np

from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image as process_image
from keras.utils import Sequence
from keras.layers import GlobalAveragePooling2D
from keras import Model

class DeepModel():
    '''MobileNet deep model.'''
    def __init__(self):
        self._model = self._define_model()

        print('Loading MobileNet.')
        print()

    @staticmethod
    def _define_model(output_layer=-1):
        '''Define a pre-trained MobileNet model.

        Args:
            output_layer: the number of layer that output.

        Returns:
            Class of keras model with weights.
        '''
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = base_model.layers[output_layer].output
        output = GlobalAveragePooling2D()(output)
        model = Model(inputs=base_model.input, outputs=output)
        return model

    @staticmethod
    def preprocess_image(path):
        '''Process an image to numpy array.

        Args:
            path: the path of the image.

        Returns:
            Numpy array of the image.
        '''
        img = process_image.load_img(path, target_size=(224, 224))
        x = process_image.img_to_array(img)
        x = preprocess_input(x)
        return x

    @staticmethod
    def cosine_distance(input1, input2):
        '''Calculating the distance of two inputs.

        The return values lie in [-1, 1]. `-1` denotes two features are the most unlike,
        `1` denotes they are the most similar.

        Args:
            input1, input2: two input numpy arrays.

        Returns:
            Element-wise cosine distances of two inputs.
        '''
        return np.dot(input1, input2.T) / \
               np.dot(np.linalg.norm(input1, axis=1, keepdims=True), \
                      np.linalg.norm(input2.T, axis=0, keepdims=True))

    def extract_feature(self, generator):
        '''Extract deep feature using the MobileNet model.

        Args:
            generator: a prediction generator inherited from `keras.utils.Sequence`.

        Returns:
            Flattened feature vectors of all inputs in the batch.
        '''
        # Get the raw features from the model's prediction
        features = self._model.predict(generator)

        # Debugging: Print the shape of the extracted features before flattening
        print("Raw feature shape before flattening:", features.shape)

        # Flatten each feature vector for each image in the batch
        flattened_features = features.reshape(features.shape[0], -1)

        # Debugging: Print the shape of the flattened features
        print("Flattened feature shape:", flattened_features.shape)

        return flattened_features


class DataSequence(Sequence):
    '''Predict generator inherited from `keras.utils.Sequence`.'''
    def __init__(self, paras, generation, batch_size=32):
        self.list_of_label_fields = []
        self.list_of_paras = paras
        self.data_generation = generation
        self.batch_size = batch_size
        self.__idx = 0

    def __len__(self):
        '''The number of batches per epoch.'''
        return int(np.ceil(len(self.list_of_paras) / self.batch_size))

    def __getitem__(self, idx):
        '''Generate one batch of data.'''
        paras = self.list_of_paras[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x, batch_fields = self.data_generation(paras)

        if idx == self.__idx:
            self.list_of_label_fields += batch_fields
            self.__idx += 1

        return np.array(batch_x)
