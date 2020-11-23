
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
        
        self.label_names = ["turn left", "turn right", "jump", "slide", "lean left", "lean right"]
        
        # feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        # base_layer = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

    # Create the model
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


    
# Train the model
def train(self, image_shape, epochs, train_data, train_labels):
    # Compile the model
    self.create_model(image_shape).compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                   
    # And train it
    return self.model.fit(train_data, train_labels, epochs = epochs);
    
# Predict the output of a single image (or a bunch) in a stream
def insta_predict(self, test_img, labels):
    # For larger batches
    #preds = self.model.predict(test_img);
    
    # Make sure the test_img is 4-D
    try:
        x = test_img.shape[3]
        x = test_img
    except:
        x = tf.expand_dims(test_img, axis=0);
    
    # For a single batch
    preds = self.model(x, training=False);
    
    # Get the labels from the predictions
    labels = []
    for i in preds:
        labels.append(self.label_names(tf.math.argmax(i)))
                      
    return labels
    
def main():
    # Get the data
    batch_size = ?
    image_shape = [384, 720, 3]
    
    # PREPROCESS.PY
    
    # BATCH UP THE DATA
    
    #Create a model object
    self = Model(batch_size, num_classes)
    
    # train the model
    print("\nStarting Training...")
    model = train(self, image_shape, epochs, train_data, train_labels)
    
    # Call insta_predict -> returns a 1-D list of the labels of the images put in
    # labels = insta_predict(self, test_img)
    
    # See the test results
    # test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2, batch_size=batch_size)

    return labels


if __name__ == '__main__':
    main()
