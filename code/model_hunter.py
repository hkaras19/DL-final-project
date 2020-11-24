from preprocess import get_data
import tensorflow as tf
from keras.applications.resnet50 import ResNet50

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        
        # hyperparamters
        self.batch_size = 128
        self.num_classes = ?
        self.image_shape = ?

        self.actions = []

        # create the model
        self.base_model = ResNet50(input_shape=self.image_shape, include_top=False, weights='imagenet')
        self.base_model.trainable = False
        self.pool_layer = tf.keras.layers.MaxPooling2D()
        self.out_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')
        self.model = tf.keras.Sequential([self.base_model, self.pool_layer, self.out_layer])

    def get_prediction(self, inputs):
        logits = self.model(inputs)
        action = 
        return action

    def train(self, epochs, batch_size, x_train, y_train):
        print("Training...")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.fit(
            x=x_train, 
            y=y_train,
            epochs=epochs,
            batch_size=batch_size)

    def test(self, batch_size, x_test, y_test):
        print("Evaluate on test data")
        results = self.model.evaluate(
            x=x_test, 
            y=y_test, 
            batch_size=batch_size)

        print("test loss, test acc:", results)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_data()
    model = Model()
    model.train(1, 128, x_train, y_train)
    model.test(128, x_test, y_test)
