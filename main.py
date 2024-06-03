# Main function
def main():
    #imports
    import os
    from sortByMood import sort_files_by_mood
    from h5converterV2 import convert_audio_to_spectrogram
    from splitFiles import split_files

    #source directory
    source_directory = '/home/user/VQ-MAE-S-code/config_speech_vqvae/dataset/Audio_Speech_Actors_01-24'
    target_directory = '/home/user/VQ-MAE-S-code/config_speech_vqvae/dataset/spectrograms'
    target_directory_sorted = '/home/user/VQ-MAE-S-code/config_speech_vqvae/dataset/spectrograms_sortedByMood'
    train_directory = '/home/user/VQ-MAE-S-code/config_speech_vqvae/dataset/train'
    test_directory = '/home/user/VQ-MAE-S-code/config_speech_vqvae/dataset/test'

    #convert to spectrograms
    if len(os.listdir(target_directory)) == 0:
        print('Converting audio to spectrograms...')
        convert_audio_to_spectrogram(source_directory, target_directory)

    #organize files by mood
    print('Organizing files by mood...')
    sort_files_by_mood(target_directory, target_directory_sorted)

    #split 80/20
    print('Splitting files...')
    split_files(target_directory_sorted, train_directory, test_directory)

    #train model
    print('Training model...')
    

    #save model

    #test model


    print("end of main")

if __name__ == "__main__":
    main()