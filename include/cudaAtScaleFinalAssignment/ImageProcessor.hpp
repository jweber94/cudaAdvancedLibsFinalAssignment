#pragma once

#include <iostream>
#include <string>

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

    bool processImage(const std::string &pathToPgm)
    {
        cv::Mat image = cv::imread(pathToPgm, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            std::cerr << "ERROR: Could not load image" << std::endl;
            return false;
        }

        if (image.type() != CV_8UC1)
        {
            std::cerr << "WARNING: Image is not a CV_8UC1. The data will be converted" << std::endl;
            image.convertTo(image, CV_8UC1); // Sicherstellen, dass es 8-Bit, 1-Kanal ist
        }

        std::cout << "Cols: " << image.cols << ", Height: " << image.rows << std::endl;

        
        return true;
    }

    ImageProcessor() = delete;

private:
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