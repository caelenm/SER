import unittest
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class TestTrainScript(unittest.TestCase):
    def setUp(self):
        self.spectrograms_dir = '/home/user/VQ-MAE-S-code/config_speech_vqvae/dataset/spectrograms'
        self.target_size = (150, 150)
        self.batch_size = 32

    def test_data_split(self):
        train_test_split_percentage = 0.8
        train_data, test_data = train_test_split(os.listdir(self.spectrograms_dir), test_size=1-train_test_split_percentage, random_state=42)
        self.assertEqual(len(train_data), int(len(os.listdir(self.spectrograms_dir)) * train_test_split_percentage))
        self.assertEqual(len(test_data), int(len(os.listdir(self.spectrograms_dir)) * (1 - train_test_split_percentage)))

    def test_data_generator(self):
        datagen = ImageDataGenerator(rescale=1./255)
        train_generator = datagen.flow_from_directory(
            self.spectrograms_dir,
            target_size=self.target_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            subset='training'
        )
        test_generator = datagen.flow_from_directory(
            self.spectrograms_dir,
            target_size=self.target_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            subset='validation'
        )
        self.assertEqual(train_generator.n, int(len(os.listdir(self.spectrograms_dir)) * 0.8))
        self.assertEqual(test_generator.n, int(len(os.listdir(self.spectrograms_dir)) * 0.2))
        self.assertEqual(train_generator.batch_size, self.batch_size)
        self.assertEqual(test_generator.batch_size, self.batch_size)

    def test_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.target_size[0], self.target_size[1], 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(os.listdir(self.spectrograms_dir)), activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.assertEqual(len(model.layers), 7)
        self.assertEqual(model.input_shape, (None, self.target_size[0], self.target_size[1], 3))
        self.assertEqual(model.output_shape, (None, len(os.listdir(self.spectrograms_dir))))

if __name__ == '__main__':
    unittest.main()