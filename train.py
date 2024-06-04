import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set the path to the spectrograms directory
#spectrograms_dir = '/home/user/VQ-MAE-S-code/config_speech_vqvae/dataset/spectrograms'

# Check if the directory is empty
# if not os.listdir(spectrograms_dir):
#     raise ValueError(f"The directory {spectrograms_dir} is empty. Please check the path.")

# # Split the data into training and testing sets (80% training, 20% testing)
# train_test_split_percentage = 0.8
# train_data, test_data = train_test_split(os.listdir(spectrograms_dir), test_size=1-train_test_split_percentage, random_state=42)


def train(train_data_dir, test_data_dir):

    # Create an ImageDataGenerator to preprocess the data
    datagen = ImageDataGenerator(rescale=1./255)

    # Define the target size and batch size for the generator
    target_size = (150, 150)
    batch_size = 32

    # Generate training and testing data using the ImageDataGenerator
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size
        #subset='training'
    )

    test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size
        #subset='validation'
    )

    # Build the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator,
        verbose=2
    )

    # Save the trained model to a file
    import datetime
    now = datetime.datetime.now()
    model_path = f'trained/model_{now.strftime("%Y%m%d%H%M%S")}.tf'
    model.save(model_path)
    print(f'Model saved to {model_path}')

    return model_path
