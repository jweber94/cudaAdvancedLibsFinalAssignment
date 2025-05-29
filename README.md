# [Negative Image Generator](https://en.wikipedia.org/wiki/Negative_(photography)) using NVIDIA NPP with CUDA

## Overview

This project demonstrates the use of NVIDIA Performance Primitives (NPP) library with CUDA to perform the calculation of negative images. The goal is to utilize GPU acceleration to efficiently calculate the inverse of grey-value images, leveraging the computational power of modern GPUs. The project is a part of the CUDA at Scale for the Enterprise course and serves as a template for understanding how to implement basic image processing operations using CUDA and NPP in a CMake based build system.

To show off the capabilities of the system, the sub-dataset [_miscellaneous_](https://sipi.usc.edu/database/database.php?volume=misc) part of the [USC Viterbi School of Engineering's SIPI Image Database](https://sipi.usc.edu/database/database.php) is used. Since the Coursera Lab-Environments are not capable of `wget` or `curl` from the databases website, the data can be found within the `./output` folder.

The system is capable of generating negative images from all `*.tiff` images in a given folder. See the `run.sh` on how to use other folders as a source where it should transform all `*.tiff` images.

Calculating the negative of an image is an interesting visual effect. Here is an example based on the image 4.2.06 of the miscellanious dataset.

Original:

![image](./data/exampleResult/original.png)

Result of the negative algorithm:

![image](./data/exampleResult/negative.png)

## Code Organization
```build/```
This folder should hold all binary/executable and cmake-generated code that is built automatically or manually.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```src/```
Here you can find the source code. `main.cu` is the entrypoint to the C++/Cuda application.

```include/```
Here are the include files of the project. It is best practice to split between src und include in your repository.
Be aware that we have a subfolder `include/cudaAtScaleFinalAssignment/` where the project includes are placed, whereas `CLI11.h` (a header based command line parsing library) is directly placed under the include folder.

```thirdparty/```
This is the folder where third party software (in this case the cuda-samples repository) is placed as git submodules.

```helpers/```
This is a folder with helper scripts to make the data handling more easy.

```output/```
This is (by the time you are cloning the repository) an empty folder that is used to save the results of the algorithm.

```INSTALL.sh```
***DO NOT CALL THIS ON YOUR OWN IF YOU DO NOT KNOW WHAT YOU ARE DOING.*** This script sets up the environment such that the upcoming programs and scirpts can run properly. It will be invoked implicitly by the `Makefile`

```Makefile```
This is the script which should be used by the user of this repository.

```CMakeLists.txt```
This is the CMake build script on which the Cuda application is build.

```run.sh```
***DO NOT CALL THIS ON YOUR OWN IF YOU DO NOT KNOW WHAT YOU ARE DOING.*** This will run all programs. It will be invoked by the `Makefile`

## Key Concepts
This main point of the project is to use Cuda in combination with a proper CMake build system, which is basically the industry standart nowadays.

Also: (Very) Basic Image Processing and how to use the NPP Library. 

## Supported SM Architectures
[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes
Linux

## Supported CPU Architecture
x86_64

## CUDA APIs involved
* Cuda Runtime API
* NVidia NPP library
* cuda-samples API

## Dependencies needed to build/run
* [FreeImage](https://github.com/danoli3/FreeImage) needs to be installed on your linux system in a system wide manner
* [NPP](https://docs.nvidia.com/cuda/npp/) which comes automatically with the cuda toolkit, so if you have the cuda toolkit installed, everything will be fine 
* [cuda-samples](https://github.com/NVIDIA/cuda-samples) which comes with the submodules of this repository

## Prerequisites
Download and install the [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run
First you have to download the repository on a linux machine on which the above mentioned software is installed. This is done by:

* ```$ git clone https://github.com/jweber94/cudaAtScaleFinalAssignment.git```

You have to add the needed submodules by going into the project root folder and clone the submodules with git

* ```$ cd cudaAtScaleFinalAssignment && git submodule update --init --recursive ```

Now you are able to use the `Makefile` to use the algorithm. By calling the _all_ target, everything will be executed on its own:

* ```$ make all```

If you want the system under your own control, feel free to inspect the `Makefile` to see whats happening in the background.
