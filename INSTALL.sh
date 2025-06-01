#!/usr/bin/bash

pip install -r helpers/requirements.txt
echo "Giving the python scrpts the correct access rights to run"
chmod a+x ./helpers/convertToP*
echo "Installation and data setup finished"