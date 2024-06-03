import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tqdm
#AI tools were used to help create the following code

# Path to the input audio files
audio_files_dir = '/home/user/VQ-MAE-S-code/config_speech_vqvae/dataset/Audio_Speech_Actors_01-24'

# Path to the output spectrograms
spectrograms_dir = '/home/user/VQ-MAE-S-code/config_speech_vqvae/dataset/spectrograms'


def convert_audio_to_spectrogram(audio_files_dir, spectrograms_dir):
    # Ensure the output directory exists
    os.makedirs(spectrograms_dir, exist_ok=True)

    # Get a list of all .wav files in the audio files directory
    wav_files = [f for f in os.listdir(audio_files_dir) if f.endswith('.wav')]

    pbar = tqdm.tqdm(total=len(wav_files), desc="Processing audio files")

    # Loop over each directory, subdirectory, and file in the audio files directory
    for dirpath, dirnames, filenames in os.walk(audio_files_dir):
        for filename in filenames:
            if filename.endswith('.wav'):
                # Construct the full path to the audio file
                audio_file = os.path.join(dirpath, filename)

                # Load the audio file
                audio, sr = librosa.load(audio_file)

                # Compute the spectrogram
                spectrogram = np.abs(librosa.stft(audio, n_fft=2048))

                # Convert the spectrogram to dB scale
                spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

                # Plot the spectrogram
                plt.figure(figsize=(10, 6))
                plt.imshow(spectrogram_db, aspect='auto', origin='lower')

                # Turn off the axis
                plt.axis('off')

                # Construct the full path to the output .svg file
                svg_file = os.path.join(spectrograms_dir, os.path.splitext(filename)[0] + '.svg')

                # Save the plot to the spectrograms directory
                plt.savefig(svg_file, format='svg', bbox_inches='tight', pad_inches=0)

                pbar.update(1)

                # Close the plot to free up memory
                plt.close()
    return