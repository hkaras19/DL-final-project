
from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data

import os
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self, batch_size, num_classes):
        super(Model, self, batch_size, num_classes).__init__()
        
        # Hyperparamters
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        # base_layer = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

    def create_model(image_shape):
        # Function to create the transfer-learned model used for training.
        # Inputs: train_data, test_data - the data used for training and testing - # Batch_size, x, y, z

        # INPUT_SHAPE VARIABLE subect TO CHANGE
        # Layers
        base_model = tf.keras.applications.ResNet50(input_shape=image_shape,
                                                         include_top=False,
                                                         weights='imagenet')
        base_model.trainable = False
        pool_layer = tf.keras.layers.MaxPooling2D();
        out_layer = tf.keras.layers.Dense(self.num_classes)
        
        # 2D convolution to recognize the key aspects in the image
        
        #Make the model
        return tf.keras.Sequential([base_model, pool_layer, out_layer])

def train(self, image_shape, epochs, train_data, train_labels):
    # Compile the model
    self.create_model(image_shape).compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                   
    # And train it
    return self.model.fit(train_data, epochs = epochs);
    
def main():
    # Get the data
    
    # PREPROCESS.PY
    
    # BATCH UP THE DATA
    
    #Create a model object
    self = Model(batch_size, num_classes)
    
    # train the model
    model = train(self, image_shape, epochs, train_data, train_labels)
    
    return model


if __name__ == '__main__':
    main()
