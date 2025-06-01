#!/usr/bin/python3
import os
import sys
from PIL import Image
import numpy as np

def jpg_folder_to_pgm(input_folder):
    """
    Converts all JPG files in a folder to PGM (Portable Graymap) files.

    Args:
        input_folder (str): The path to the input folder containing JPG files.
    """
    if not os.path.isdir(input_folder):
        print(f"Error: The specified path '{input_folder}' is not a valid folder.")
        return

    jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpeg', '.jpg'))]

    if not jpg_files:
        print(f"No JPG files found in the folder '{input_folder}'.")
        return

    print(f"Starting conversion of {len(jpg_files)} JPG files in the folder '{input_folder}'.")

    for jpg_file_name in jpg_files:
        jpg_path = os.path.join(input_folder, jpg_file_name)
        base_name, _ = os.path.splitext(jpg_file_name)
        pgm_file_name = base_name + ".pgm"
        pgm_path = os.path.join(input_folder, pgm_file_name)

        try:
            # Open the JPG image
            img = Image.open(jpg_path)

            # Convert to grayscale if it's not already grayscale
            if img.mode != 'L':
                img = img.convert('L')

            # Get image data as a NumPy array
            image_array = np.array(img)
            height, width = image_array.shape

            # Write the PGM file in binary format
            with open(pgm_path, 'wb') as pgm_file:
                # Write the PGM header
                pgm_file.write(b'P5\n')  # Magic number for binary PGM
                pgm_file.write(f'{width} {height}\n'.encode('ascii'))
                pgm_file.write(b'255\n')  # Maximum grayscale value

                # Write the image data
                pgm_file.write(image_array.tobytes())

            print(f"Converted: '{jpg_file_name}' -> '{pgm_file_name}'")

        except FileNotFoundError:
            print(f"Error: The file '{jpg_path}' was not found (should not happen).")
        except Exception as e:
            print(f"Error during conversion of '{jpg_file_name}': {e}")

    print("Conversion of all JPG files completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <path_to_jpg_folder>")
        sys.exit(1)

    input_folder_path = sys.argv[1]
    jpg_folder_to_pgm(input_folder_path)