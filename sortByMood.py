import os
import shutil

def sort_files_by_mood(source_directory, target_directory_sorted):
    # Define mood dictionary
    mood_dict = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    # Iterate over files in the specified directory
    for filename in os.listdir(source_directory):
        if filename.endswith('.png'):
            # Extract the mood from the filename
            mood_number = filename.split('-')[2]

            # Get the mood name from the dictionary
            mood_name = mood_dict.get(mood_number, 'unknown')

            # Create a new directory for this mood if it doesn't exist
            new_directory = os.path.join(target_directory_sorted, mood_name)
            os.makedirs(new_directory, exist_ok=True)

            # Move the file into the new directory
            shutil.move(os.path.join(source_directory, filename), os.path.join(new_directory, filename))