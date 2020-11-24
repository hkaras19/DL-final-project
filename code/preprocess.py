import tensorflow as tf

def get_data(img_height, img_width, batch_size):
    """
    Purpose: create the training and test datasets
    Args: image dimensions and batch size
    Return: train and test objects
    """

    data_dir = '../data' # image directory

    # loads all of the images from dir and figures out their labels

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # optimizing for disk
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    # repeat for testing set
    
    return train_ds