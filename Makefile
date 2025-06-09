build:
	mkdir build
	cmake -Bbuild -S.
	cmake --build ./build/

run:
	/bin/bash run.sh

clean:
	-rm -f ./output/*
	-rm -rf build

all: clean build run