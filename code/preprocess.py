import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def get_data(img_height, img_width, batch_size):
    """
    Purpose: create the training and test datasets
    Args: image dimensions and batch size
    Return: train and test objects
    """

    data_dir = '../data' # image directory

    # loads all of the images from dir and figures out their labels

#    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#        data_dir,
#        seed=123,
#        image_size=(img_height, img_width),
#        batch_size=batch_size)
    
   # print(train_ds)
   
   
   # Loads all of the images from data_dir into a dictionary
    builder = tfds.ImageFolder(data_dir, shape = (img_height, img_width, 3))
    print(builder.info)

    # Get the Dataset from the dictionary
    ds = builder.as_dataset(split='train', shuffle_files=True)
    dataset = builder.as_dataset(as_supervised=True)['train']
    
    # Gets 1363 images
    # train_aye = np.zeros((1363, img_height, img_width, 3))
    # train, labels = map(list,zip(*dataset))
    # real_train = np.copyto(train_aye, train)
    # train = np.fromiter(train, np.float32)
    # train = np.ndarray(buffer=train, dtype=np.uint8, shape=(1363, img_height, img_width, 3))
    
    # I think this is inefficient: for loops to fill train with dataset of size 100 for testing purposes
    # But...I've tried stuff like above and its much faster than all of that
    i = 0
    train = np.zeros((1363, 720, 384, 3))
    labels = np.zeros((1363))
    
    for item in dataset:
        train[i] = item[0]
        labels[i] = item[1]
        i += 1

    labels = tf.one_hot(labels, 6)
    
    return train, labels
