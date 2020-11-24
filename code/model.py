from preprocess import get_data
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import matplotlib as plt
import numpy as np
from tensorflow.keras.preprocessing import image

class Model(tf.keras.Model):
    def __init__(self, is_new):
        super(Model, self).__init__()
        """
        This model will be used to classify temple run images into the action that should 
        be taken at this point in the game.
        """

        print("Setting up model...")
        
        # hyperparamters

        self.batch_size = 20
        self.num_classes = 6
        self.img_height = 720
        self.img_width = 384
        self.image_shape = (self.img_height, self.img_width, 3)
        self.class_names = ['jmp', 'lean_left', 'lean_right', 'slide', 'turn_left', 'turn_right']

        # create the model

        if is_new: # check if making a new model or loading old (trained) model
            self.normalize = tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=self.image_shape) # normalizing layer
            self.base_model = ResNet50(input_shape=self.image_shape, include_top=False, weights='imagenet') # resnet50
            self.base_model.trainable = False # freeze the base layer
            self.pool_layer = tf.keras.layers.GlobalAveragePooling2D()
            self.out_layer = tf.keras.layers.Dense(self.num_classes, activation='relu')
            self.model = tf.keras.Sequential([self.base_model, self.pool_layer, self.out_layer]) # final model

        else:
            self.model = tf.keras.models.load_model('resnet50_model') # load the model from saved version

    def get_prediction(self, img, label=None):
        """
        Purpose: predict an action given a game screenshot
        Args: img is the screenshot to be examined
        Return: the action to take
        """

        print("Getting prediction...")
        img_array = tf.keras.preprocessing.image.img_to_array(img) # convert image to array
        img_array = tf.expand_dims(img_array, 0) # create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) # get logits
        action = self.class_names[np.argmax(score)]

        # print(
        #     "This image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        # )

        return action

    def train(self, epochs, train_data, train_labels):
        """
        Purpose: train the network to recognize actions based on images
        Args: number of epochs to train for, keras training dataset object
        Return: nothing
        """

        print("Training...")
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # set up optimizer and loss

        print(self.model.summary())

        # train the model
        history = self.model.fit(
            train_data,
            train_labels,
            batch_size = self.batch_size,
            verbose=2,
            epochs=epochs)
        
        self.model.save('resnet50_model') # save the model weights
        print('Saved model to disk')

        visualize_results(history, epochs) # look at results

def visualize_results(history, epochs):
    """
    Purpose: look at the results from training
    Args: history is the metrics from training, epochs is the number of epochs trained on
    Return: nothing
    """

    acc = history.history['accuracy'] # get the training data and plot graphs
    loss = history.history['loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.legend(loc='upper right')
    plt.title('Training Loss')
    plt.show()


if __name__ == '__main__':
    is_new = False
    model = Model(is_new)
    
    if is_new:
        train_data, train_labels = get_data(model.img_height, model.img_width, model.batch_size)
        model.train(10, train_data, train_labels)
    
    filename = '../data/train/jmp/j00000.png'
    img = image.load_img(filename, target_size=(model.img_height, model.img_width))
    model.get_prediction(img)
    
    
    
