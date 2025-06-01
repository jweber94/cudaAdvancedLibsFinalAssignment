#pragma once

#include <iostream>
#include <string>

#include <cufft.h>

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_arithmetic_and_logical_operations.h>

#include "UtilNPP/ImageIO.h"
#include "UtilNPP/ImagesCPU.h"
#include "UtilNPP/ImagesNPP.h"
#include <opencv2/opencv.hpp>

class ImageProcessor
{
public:
    ImageProcessor(const std::string &processingOutputFolder) : m_outputFolder{processingOutputFolder}
    {
        // check if GPU exists to run the algorithm on it
        cudaGetDeviceCount(&m_deviceCount);
        if (m_deviceCount == 0)
        {
            std::cerr << "No CUDA capable devices found! The program can not work without a GPU" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "Devices Found - count is: " << m_deviceCount << " using device No. 0" << std::endl;
        cudaSetDevice(0);
    }

    bool processImage(const std::string &pathToImg)
    {
        // read image to RAM
        cv::Mat img = readImage(pathToImg);

        // Copy image data to GPU
        auto pImgGPU = copyToGpu(img);
        if (nullptr == pImgGPU)
        {
            std::cerr << "Could not copy the image to GPU" << std::endl;
            return false;
        }

        // Do Fourrier Transform

        // Calculate Cross-Correllation

        // Convert Corellation Matrix back

        // Copy Correllation Matrix to CPU

        // Search for maximum (e.g. with thrust lib)

        // Free memory on GPU
        cudaFree(pImgGPU);

        return true;
    }

    ImageProcessor() = delete;

private:
    cv::Mat readImage(const std::string &pathToJpg)
    {
        cv::Mat image = cv::imread(pathToJpg, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            std::cerr << "ERROR: Could not load image" << std::endl;
            return {};
        }
        if (image.type() != CV_8UC1)
        {
            std::cerr << "WARNING: Image is not a CV_8UC1. The data will be converted" << std::endl;
            image.convertTo(image, CV_8UC1); // Sicherstellen, dass es 8-Bit, 1-Kanal ist
        }
        std::cout << "Cols: " << image.cols << ", Height: " << image.rows << std::endl;
        return image;
    }

    cufftComplex *copyToGpu(const cv::Mat &imgData)
    {
        // sanity check
        if (imgData.empty() || imgData.type() != CV_8UC1)
        {
            std::cerr << "ERROR: Input image must be a non-empty CV_8UC1 (8-bit grayscale) image." << std::endl;
            return nullptr;
        }
        if (!imgData.isContinuous())
        {
            std::cerr << "ERROR: Could not copy non-continous memory from CPU to GPU - stop processing" << std::endl;
            return nullptr; // Vereinfachung für dieses Beispiel
        }

        // convert greyscale data to cufftComplex
        size_t num_pixels = imgData.total();
        std::vector<cufftComplex> host_complex_buffer(num_pixels);
        const uint8_t *img_ptr = imgData.ptr<uint8_t>(0); // get pointer to the raw greyscale data on CPU cv::Mat
        for (size_t i = 0; i < num_pixels; ++i)
        {
            host_complex_buffer[i].x = static_cast<float>(img_ptr[i]); // Real part from grayscale
            host_complex_buffer[i].y = 0.0f;                           // Imaginary part is zero for real input
        }

        // allocate memory on device/GPU
        cufftComplex *d_complex_image = nullptr;
        std::size_t complex_image_bytes = num_pixels * sizeof(cufftComplex);
        cudaError_t err = cudaMalloc((void **)&d_complex_image, complex_image_bytes);
        if (cudaError::cudaSuccess != err)
        {
            std::cerr << "Unable to allocate memory on GPU" << std::endl;
            return nullptr;
        }

        err = cudaMemcpy(d_complex_image, host_complex_buffer.data(), complex_image_bytes, cudaMemcpyHostToDevice);
        if (cudaError::cudaSuccess != err)
        {
            std::cerr << "Unable to allocate memory on GPU" << std::endl;
            cudaFree(d_complex_image); // avoid memory leak on GPU
            return nullptr;
        }

        return d_complex_image;
    }

    std::string generateNewNameFromPath(const std::string &path)
    {
        size_t lastSlash = path.rfind('/');
        std::string nameWithPath = (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);
        size_t lastDot = nameWithPath.rfind('.');
        std::string nameWithoutEnding;
        std::string ending;
        if (lastDot != std::string::npos)
        {
            nameWithoutEnding = nameWithPath.substr(0, lastDot);
            ending = nameWithPath.substr(lastDot);
        }
        else
        {
            nameWithoutEnding = nameWithPath;
        }
        std::string newName = nameWithoutEnding + "_negative";
        newName += ending;
        return newName;
    }

    std::string m_outputFolder;
    int m_deviceCount{0};
};