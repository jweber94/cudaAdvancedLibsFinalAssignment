cmake_minimum_required(VERSION 3.16)

## basic cmake setup
project(cudaAtScaleFinalAssignment LANGUAGES CXX CUDA) # enable cuda libraries is easy since cmake 3.10 - this includes the cuda runtime api but not the additional libraries of cuda https://stackoverflow.com/questions/66327073/how-to-find-and-link-cuda-libraries-using-cmake-3-15-and-later

set(CMAKE_CXX_STANDARD 20)

## create the executable
set(SRCS
    src/main.cu
)

set(INCLUDES
    include/CLI11.h
    include/cudaAtScaleFinalAssignment/ImageProcessor.hpp
    include/cudaAtScaleFinalAssignment/PgmDataGetter.hpp
)

add_executable(${PROJECT_NAME} ${SRCS})

## Add libraries that are needed to build the project 
# FreeImage include in order to use the convenience tooling of cuda-samples for image processing
find_library(FREEIMAGE_LIBRARY
    NAMES freeimage
    PATHS /usr/lib/x86_64-linux-gnu/
    NO_DEFAULT_PATH
)
find_path(FREEIMAGE_INCLUDE_DIR
    NAMES FreeImage.h
    PATHS /usr/include/
    NO_DEFAULT_PATH
)

# Add cuda NPPI - the third party cuda libs need to be added additionally to cmake 
find_library(NPPI_LIBRARY
    NAMES nppisu
    PATHS /usr/local/cuda-11.3/targets/x86_64-linux/lib/stubs/
    NO_DEFAULT_PATH
)
find_path(NPPI_INCLUDE_DIR
    NAMES nppi.h
    PATHS /usr/local/cuda-11.3/targets/x86_64-linux/include/
    NO_DEFAULT_PATH
)

# Add cuda NPPIal (arithmetic and logical operations) as dynamic library since nppiNot_8u_C1R is part of it
find_library(NPPIAL_LIBRARY
    NAMES nppial
    PATHS /usr/local/cuda-11.3/targets/x86_64-linux/lib/stubs/
    NO_DEFAULT_PATH
)
find_path(NPPIAL_INCLUDE_DIR
    NAMES nppi_arithmetic_and_logical_operations.h
    PATHS /usr/local/cuda-11.3/targets/x86_64-linux/include/
    NO_DEFAULT_PATH
)


## building the target with dependencies
target_include_directories(${PROJECT_NAME} 
    PUBLIC include/ 
    PUBLIC include/cudaAtScaleFinalAssignment/
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PUBLIC thirdparty/cuda-samples/Common/
    PUBLIC ${FREEIMAGE_INCLUDE_DIR}
    PUBLIC ${NPPI_INCLUDE_DIR}
    PUBLIC ${NPPIAL_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC ${FREEIMAGE_LIBRARY} 
    PUBLIC ${NPPI_LIBRARY}
    PUBLIC ${NPPIAL_LIBRARY}
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>) # you can not use the normal CXX standart if you use cuda along with it