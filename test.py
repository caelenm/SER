import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import image_dataset_from_directory

def test_model(model_dir, test_dir):
    # Load the trained model
    loaded_model = tf.keras.models.load_model(model_dir)

    img_height = 150
    img_width = 150
    batch_size = 32

    test_data = image_dataset_from_directory(test_dir, image_size=(img_height, img_width), batch_size=batch_size)
    test_images = []
    test_labels = []
    for images, labels in test_data:
        test_images.append(images)
        test_labels.append(labels)

     # Get the class names from the directory names
    classes = test_data.class_names
    # Convert labels to one-hot encoding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(classes)
    test_labels = tokenizer.texts_to_matrix(test_labels)

    # Evaluate the model on the test data
    loss, accuracy = loaded_model.evaluate(test_images, test_labels)

    # Print the test accuracy
    print(f'Test accuracy: {accuracy}',f'Test loss: {loss}' )
    # Write the model name and results to a log file
    with open('log.txt', 'w') as file:
        file.write(f'Model: {model_dir}\n')
        file.write(f'Test accuracy: {accuracy}\n')
        file.write(f'Test loss: {loss}\n')

# Usage example:
# test_model('path/to/trained_model.h5', (test_images, test_labels))