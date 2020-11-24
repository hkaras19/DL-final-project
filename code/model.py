from preprocess import get_data
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
import matplotlib as plt
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, is_new):
        super(Model, self).__init__()
        """
        This model will be used to classify temple run images into the action that should 
        be taken at this point in the game.
        """
        
        # hyperparamters

        self.batch_size = 128
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
            self.pool_layer = tf.keras.layers.MaxPooling2D() # adding pooling and dense layers
            self.out_layer = tf.keras.layers.Dense(self.num_classes, activation='leaky_relu')
            self.model = tf.keras.Sequential([self.base_model, self.pool_layer, self.out_layer]) # final model
        else:
            self.model = tf.keras.models.load_model('resnet50_model') # load the model from saved version

    def get_prediction(self, img):
        """
        Purpose: predict an action given a game screenshot
        Args: img is the screenshot to be examined
        Return: the action to take
        """

        img_array = tf.keras.preprocessing.image.img_to_array(img) # convert image to array
        img_array = tf.expand_dims(img_array, 0) # create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) # get logits

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )

    def train(self, epochs, train_ds):
        """
        Purpose: train the network to recognize actions based on images
        Args: number of epochs to train for, keras training dataset object
        Return: nothing
        """

        print("Training...")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy']) # set up optimizer and loss

        # train the model
        history = self.model.fit(
            train_ds,
            epochs=epochs)
        
        self.model.save('resnet50_model') # save the model weights
        print('Saved model to disk')

        self.visualize_results(history, epochs) # look at results

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
    model = Model(is_new=True)
    train_ds = get_data(model.img_height, model.img_width, model.batch_size)
    model.train(1, train_ds)
