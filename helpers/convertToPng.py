#!/usr/bin/python3
import os
import sys
from PIL import Image
import numpy as np

def pgm_folder_to_png(input_folder):
    """
    Converts all PGM files in a folder to PNG files.

    Args:
        input_folder (str): The path to the input folder containing PGM files.
    """
    if not os.path.isdir(input_folder):
        print(f"Error: The specified path '{input_folder}' is not a valid folder.")
        return

    pgm_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pgm')]

    if not pgm_files:
        print(f"No PGM files found in the folder '{input_folder}'.")
        return

    print(f"Starting conversion of {len(pgm_files)} PGM files in the folder '{input_folder}'.")

    for pgm_file_name in pgm_files:
        pgm_path = os.path.join(input_folder, pgm_file_name)
        base_name, _ = os.path.splitext(pgm_file_name)
        png_file_name = base_name + ".png"
        png_path = os.path.join(input_folder, png_file_name)

        try:
            # Open the PGM image in binary read mode
            with open(pgm_path, 'rb') as pgm_file:
                # Read the PGM header
                magic_number = pgm_file.readline().decode('ascii').strip()
                if magic_number != 'P5':
                    print(f"Error: Invalid PGM format in '{pgm_file_name}'. Expected 'P5'.")
                    continue

                dimensions = pgm_file.readline().decode('ascii').strip().split()
                if len(dimensions) != 2:
                    print(f"Error: Invalid dimensions in PGM header of '{pgm_file_name}'.")
                    continue
                width = int(dimensions[0])
                height = int(dimensions[1])

                max_value = pgm_file.readline().decode('ascii').strip()
                if not max_value.isdigit() or int(max_value) > 255:
                    print(f"Error: Invalid max grayscale value in PGM header of '{pgm_file_name}'.")
                    continue

                # Read the image data
                image_data = np.fromfile(pgm_file, dtype=np.uint8)
                if image_data.size != width * height:
                    print(f"Error: Incomplete image data in '{pgm_file_name}'.")
                    continue

                # Reshape the data into a 2D array
                image_array = image_data.reshape((height, width))

            # Create a PIL Image from the NumPy array
            img = Image.fromarray(image_array)

            # Save the image as PNG
            img.save(png_path)

            print(f"Converted: '{pgm_file_name}' -> '{png_file_name}'")

        except FileNotFoundError:
            print(f"Error: The file '{pgm_path}' was not found (should not happen).")
        except Exception as e:
            print(f"Error during conversion of '{pgm_file_name}': {e}")

    print("Conversion of all PGM files completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <path_to_pgm_folder>")
        sys.exit(1)

    input_folder_path = sys.argv[1]
    pgm_folder_to_png(input_folder_path)