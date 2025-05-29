#pragma once

#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_arithmetic_and_logical_operations.h>

#include "UtilNPP/ImageIO.h"
#include "UtilNPP/ImagesCPU.h"
#include "UtilNPP/ImagesNPP.h"

    class ImageProcessor {
    public:
        ImageProcessor(const std::string& processingOutputFolder) :
            m_outputFolder{processingOutputFolder}
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

        bool processImage(const std::string& pathToPgm) {
            // use the cuda-samples convenient image classes and methods for a more easy data handling
            npp::ImageCPU_8u_C1 oHostSrc;
            npp::loadImage(pathToPgm, oHostSrc);
            npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
            NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()}; // ROI is the whole image - see https://docs.nvidia.com/cuda/npp/introduction.html#nppi_conventions_lb_1roi_specification for explaination
            
            // prepare result memory
            npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height); // prepare memory for the result
            npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());

            // do the processing
            NppStatus ret = nppiNot_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI);
            if (NppStatus::NPP_NO_ERROR != ret) {
                std::cerr << "Could not process image on GPU" << std::endl;
                return false;
            }

            // copy back from the GPU to CPU and save the result to disk
            oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
            std::string resultPath = m_outputFolder + "/" + generateNewNameFromPath(pathToPgm);
            saveImage(resultPath, oHostDst);

            // free the memory on GPU since it will not clean up itself by going out of scope
            nppiFree(oDeviceSrc.data());
            nppiFree(oDeviceDst.data());
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