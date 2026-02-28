import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    images, labels = load_data(sys.argv[1])
    labels = tf.keras.utils.to_categorical(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()
    model.fit(x_train, y_train, epochs=EPOCHS)
    model.evaluate(x_test,  y_test, verbose=2)

    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(data_dir):
    imagesArr = []
    labels = []
    
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                imagesArr.append(np.array(image))
                labels.append(int(folder))
                
    return (imagesArr, labels)

def get_model():
    """
    Optimized Architecture (Based on Experiment 2.9)
    - 5x5 Kernel on first Conv2D for wider initial receptive field.
    - 64 filters on second Conv2D for detailed feature extraction.
    - 0.4 Dropout for optimal regularization.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    main()
