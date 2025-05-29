#!/usr/bin/bash

echo "Extracting data"
tar -xzf ./data/misc.tar.gz -C ./data/
echo "Install python environment to make the conversion scripts runnable"
pip install -r helpers/requirements.txt
echo "Giving the python scrpts the correct access rights to run"
chmod a+x ./helpers/convertToP*
echo "Installation and data setup finished"