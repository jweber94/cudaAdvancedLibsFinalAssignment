install:
	/bin/bash INSTALL.sh

build:
	mkdir build
	cmake -Bbuild -S.
	cmake --build ./build/

run:
	/bin/bash run.sh

clean:
	-rm -f ./output/*
	-rm -rf build
	-rm -rf ./data/misc

all: clean install build run