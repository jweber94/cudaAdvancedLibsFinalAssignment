#!/usr/bin/python3
import os
import sys
from PIL import Image
import numpy as np

def tiff_folder_to_pgm(input_folder):
    """
    Converts all TIFF files in a folder to PGM (Portable Graymap) files.

    Args:
        input_folder (str): The path to the input folder containing TIFF files.
    """
    if not os.path.isdir(input_folder):
        print(f"Error: The specified path '{input_folder}' is not a valid folder.")
        return

    tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tiff', '.tif'))]

    if not tiff_files:
        print(f"No TIFF files found in the folder '{input_folder}'.")
        return

    print(f"Starting conversion of {len(tiff_files)} TIFF files in the folder '{input_folder}'.")

    for tiff_file_name in tiff_files:
        tiff_path = os.path.join(input_folder, tiff_file_name)
        base_name, _ = os.path.splitext(tiff_file_name)
        pgm_file_name = base_name + ".pgm"
        pgm_path = os.path.join(input_folder, pgm_file_name)

        try:
            # Open the TIFF image
            img = Image.open(tiff_path)

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

            print(f"Converted: '{tiff_file_name}' -> '{pgm_file_name}'")

        except FileNotFoundError:
            print(f"Error: The file '{tiff_path}' was not found (should not happen).")
        except Exception as e:
            print(f"Error during conversion of '{tiff_file_name}': {e}")

    print("Conversion of all TIFF files completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <path_to_tiff_folder>")
        sys.exit(1)

    input_folder_path = sys.argv[1]
    tiff_folder_to_pgm(input_folder_path)