# Main function
def main():
    #imports
    import os
    from sortByMood import sort_files_by_mood
    from h5converterV2 import convert_audio_to_spectrogram
    from splitFiles import split_files
    from trainV2 import train

    #source directory
    source_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\Audio_Speech_Actors_01-24'
    target_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\spectrograms'
    target_directory_sorted = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\spectrograms_sortedByMood_png'
    train_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\train_png'
    test_directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\test_png'

    #convert to spectrograms
    if (len(os.listdir(target_directory_sorted)) == 0) & (len(os.listdir(target_directory)) == 0):
        print('Converting audio to spectrograms...')
        convert_audio_to_spectrogram(source_directory, target_directory)

    #organize files by mood
    if len(os.listdir(target_directory_sorted)) == 0:
        print('Organizing files by mood...')
        sort_files_by_mood(target_directory, target_directory_sorted)

    #split 80/20
    print('Splitting files...')
    split_files(target_directory_sorted, train_directory, test_directory)

    #train model
    print('Training model...')
    train(train_directory, test_directory)
    

    #save model

    #test model


    print("end of main")
    #train()

if __name__ == "__main__":
    main()