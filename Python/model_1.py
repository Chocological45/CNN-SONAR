import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split


EPOCHS = 10             # Number of training runs
BATCH_SIZE = 10
IMG_WIDTH = 96
IMG_HEIGHT = 96
NUM_CATEGORIES = 10     # Number of different labels that we have in the dataset
TEST_SIZE = 0.2         # Train/Test dataset split ratio


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python model.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_dataset(sys.argv[1])

    # Check that we are getting the same number of images as labels
    print(len(images), len(labels))

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Check the train and test image count
    print(len(x_train), len(x_test))

    # Get a compiled neural network
    model = get_model()
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)
    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to filesystem
    #if len(sys.argv) == 3:
    #filename = 'models/model1_d.model' #sys.argv[2]
    #model.save(filename)
    #print(f"Model saved to {filename}.")


def load_dataset(dir):
    images = list()
    labels = list()

    for data_set in os.listdir(dir):
        path = os.path.join(dir, data_set)

        if os.path.isdir(path):
            print(f"Loading files from {path}...")

            for data in os.listdir(path):
                # A file check should go here for safety
                if data != '.DS_Store':
                    # Read the image file
                    img = cv2.imread(os.path.join(path, data), cv2.IMREAD_GRAYSCALE)

                    # Resize said file using interpolation
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)

                    # Reshape the image arrays for compatibility with CNN model
                    reshaped = img.reshape(IMG_WIDTH, IMG_HEIGHT, 1)

                    # Add the image and label to corresponding lists and return
                    images.append(reshaped)
                    labels.append(int(data_set))

    return images, labels


def get_model():
    model = tf.keras.models.Sequential([

        # Convolutional layer. Learn 32 filters using 5x5 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', padding='valid', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add FC layer
        tf.keras.layers.Dense(64, activation='relu'),

        # Add output layer with softmax function
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives()]
    )

    return model


if __name__ == "__main__":
    main()
