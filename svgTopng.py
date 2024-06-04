import os
from PIL import Image
import cairosvg

directory = r'C:\Users\Caelen\Documents\VQ-MAE-S-code\config_speech_vqvae\dataset\spectrograms_sortedByMood/surprised'
# Create a new directory with the same name as the original directory but ending in _png
png_directory = f"{directory}_png"
os.makedirs(png_directory, exist_ok=True)

for filename in os.listdir(directory):
    svg_path = os.path.join(directory, filename)
    png_path = os.path.join(png_directory, f"{os.path.splitext(filename)[0]}.png")
    try:
        cairosvg.svg2png(url=svg_path, write_to=png_path)
    except Exception as e:
        print(f"Error converting {svg_path} to PNG: {e}")

# Example usage

#convert_svg_to_png(target_directory)
