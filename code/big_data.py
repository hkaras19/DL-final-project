import tensorflow as tf
import os
import numpy as np
import glob

class BigDataTrainer:
    def __init__(self, model):
        print("Setting up big data trainer...")
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.img_height = model.img_height
        self.img_width = model.img_width
        self.data_dir = '../data/train/*/*.jpeg'
        self.class_names = model.class_names
        self.model = model

    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == self.class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # resize the image to the desired size
        return tf.image.resize(img, [self.img_height, self.img_width])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def configure_for_performance(self, ds):
        # ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.model.batch_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

    def train(self):
        image_count = len(glob.glob(self.data_dir))

        print("Found {} images...".format(image_count))
        print("Getting data...")
        list_ds = tf.data.Dataset.list_files(str(self.data_dir), shuffle=False)
        i = 0
        for x, y in list_ds:
            print(x, y)
            i += 1
            if i == 100:
                return


        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

        print("Splitting data...")
        val_size = int(image_count * 0.2)
        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)

        print("Processing data...")
        train_ds = train_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
        val_ds = val_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)

        print("Optimizing data...")
        train_ds = self.configure_for_performance(train_ds)
        val_ds = self.configure_for_performance(val_ds)

        self.model.train(train_ds, val_ds, 3)

        