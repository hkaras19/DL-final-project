# from code.big_data import BigDataTrainer
from big_data import BigDataTrainer
import tensorflow as tf
#from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB0
import matplotlib as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os
#from tensorflow.keras.preprocessing import image

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

           # self.base_model = ResNet50(input_shape=self.image_shape, include_top=False, weights='imagenet') # resnet50
            self.base_model = EfficientNetB0(input_shape=self.image_shape, include_top=False, weights='imagenet') # resnet50
            self.base_model.trainable = False # freeze the base layer
            self.pool_layer = tf.keras.layers.GlobalAveragePooling2D()
            self.out_layer = tf.keras.layers.Dense(self.num_classes, activation='relu')
            self.model = tf.keras.Sequential([self.base_model, self.pool_layer, self.out_layer]) # final model
            print(self.model.summary())

        else:
            self.model = tf.keras.models.load_model('../templeflow_model_enet') # load the model from saved version
            print(self.model.summary())

    def call(self, img, label=None):
        """
        Purpose: predict an action given a game screenshot
        Args: img is the screenshot to be examined
        Return: the action to take
        """

        print("\nGetting prediction...")
        img_array = tf.keras.preprocessing.image.img_to_array(img)[:,:,:3] # convert image to array (720, 384, 4)
        img_array = tf.expand_dims(img_array, 0) # create a batch

        predictions = self.model.predict(img_array)
        # print(predictions)

        score = tf.nn.softmax(predictions[0]) # get logits
        # print(score)
        action = self.class_names[np.argmax(score)]

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence.\n"
            .format(action, 100 * np.max(score))
        )

        return action

    def train(self, train_ds, epochs):
        """
        Purpose: train the network to recognize actions based on images
        Args: number of epochs to train for, keras training dataset object
        Return: nothing
        """

        print("Training...")
        
        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']) # set up optimizer and loss

        print(tf.data.experimental.cardinality(train_ds).numpy())
        # for image, label in train_ds.take(5):
        #   print("Image shape: ", image.numpy().shape)
        #   print("Label: ", label.numpy())
        # return
        
        # train the model
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            verbose=1
        )
        
      #  self.model.save('templeflow_model_enet') # save the model weights
        print('Saved model to disk')

    #    visualize_results(history, epochs) # look at results

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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    is_new = False
    model = Model(is_new)
    
    if is_new:
        BigDataObj = BigDataTrainer(model)
        BigDataObj.train()

    images = [
        'jmp/jmp0.jpeg', 
        'lean_left/lean_left0.jpeg',
        'lean_right/lean_right0.jpeg',
        'slide/slide0.jpeg',
        'turn_left0/turn_left0.jpeg',
        'turn_right0/turn_right0.jpeg'
    ]

    # for image in images:  
    filename = '../data/train/slide/slide0.jpeg'  
    img = image.load_img(filename, target_size=(model.img_height, model.img_width))
    model(img)


