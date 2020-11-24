import tensorflow as tf

def main():
    batch_size = 128
    img_height = 720
    img_width = 384

    data_dir = '../data/'

    train_ds = tf.keras.preprocessing.image.image_dataset_from_directory(
        data_dir,
        seed=123,
        labels='inferred', # Infers labels from directory structure
        image_size=(img_height, img_width),
        batch_size=batch_size)

    print(train_ds)
    
    class_names = train_ds.class_names
    print(class_names)
    
#    for image_batch, labels_batch in train_ds:
#        print(image_batch[0])
#        print(image_batch.shape)
#        print(labels_batch.shape)
#        break

if __name__ == '__main__':
    main()
