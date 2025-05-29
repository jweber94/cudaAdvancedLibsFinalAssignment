#!/usr/bin/bash

echo "Convert the dataset to pgm data to have a better handle on it in the code"
./helpers/convertToPgm.py ./data/misc

echo "Run the application"
./build/cudaAtScaleFinalAssignment -p $(pwd)/data/misc -o $(pwd)/output

echo "Convert the resulting pgm data back to png to have them visable"
./helpers/convertToPng.py ./output