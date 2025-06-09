#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <cufft.h>

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

//#define FIREDATASET
//#define DEBUG

// forward declaration of the calculation kernel
__global__ void crossCorrelationKernel(cufftComplex *F1_gpu, cufftComplex *F2_gpu, cufftComplex *P_gpu, int rows, int complex_output_cols);
__global__ void scale_kernel(float *data, int num_elements, float scale_factor);

struct ProcessingResult {
    std::string path1;
    std::string path2;
    int xShift;
    int yShift;
    double peakIntensitiy;

    // Constructor needed for efficient emplacement later on - POD structs are not eligible to be used with C++11 emplace_back
    ProcessingResult(std::string p1, std::string p2, int xS, int yS, double pI)
        : path1(std::move(p1)), // std::move für std::string Parameter, um Kopien zu vermeiden
          path2(std::move(p2)),
          xShift(xS),
          yShift(yS),
          peakIntensitiy(pI)
    {}
};

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
        m_calculationResults.reserve(100); // make the vector big enough for a small dataset to avoid unneeded reallocation and copying of memory chunks during runtime
    }

    ~ImageProcessor() {
        // Save result to file
        std::string retPath = m_outputFolder + "crosspowerspectral_shift_results.csv";
        std::ofstream oss(retPath);
        std::size_t counter{0};
        for (auto &it : m_calculationResults)
        {
            oss << counter++ << "," << it.path1 << "," << it.path2 << "," << it.xShift << "," << it.yShift << "," << it.peakIntensitiy << std::endl;
        }
        oss.close();
    }

    bool processImage(std::string &pathToImg1, std::string &pathToImg2)
    {
        // read image to RAM
        cv::Mat img1 = readImage(pathToImg1);
        cv::Mat img2 = readImage(pathToImg2);

        if ((img1.rows != img2.rows) || (img1.cols != img2.cols)) {
            std::cerr << "Invalid image dimension" << std::endl;
            return false;
        }

        // Copy image data to GPU in real number form - cufftreal_t* or float* needed to be used with cufft
        auto pImgGPU1 = copyToGpu(img1);
        if (nullptr == pImgGPU1)
        {
            std::cerr << "Could not copy the image to GPU" << std::endl;
            return false;
        }
        auto pImgGPU2 = copyToGpu(img2);
        if (nullptr == pImgGPU2)
        {
            std::cerr << "Could not copy the image to GPU" << std::endl;
            cudaFree(pImgGPU1);
            return false;
        }

        // Do Fourrier Transform and calculate the cross correllation
        auto pCrossCorrellationMatrix = calculateCrossCorrelation(pImgGPU1, pImgGPU2, img1.rows, img1.cols);
        if (nullptr == pCrossCorrellationMatrix)
        {
            std::cerr << "Could not do the fourrier transformation" << std::endl;
            cudaFree(pCrossCorrellationMatrix);
            return false;
        }

        // Convert Corellation Matrix back
        auto pCorrellationResultReal = perform2DIFFT_and_scale(pCrossCorrellationMatrix, img1.rows, img1.cols);

        // Copy Correllation Matrix to CPU
        std::vector<float> h_ifft_correlation_result(img1.rows * img1.cols);
        auto retCpyBack =cudaMemcpy(h_ifft_correlation_result.data(), pCorrellationResultReal, img1.rows * img1.cols * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaError::cudaSuccess != retCpyBack) {
            std::cerr << "Could not copy the cross correllation matrix back to CPU" << std::endl;
            cudaFree(pCrossCorrellationMatrix);
            cudaFree(pCorrellationResultReal);
            return false;
        }

        cv::Mat correlation_mat(img1.rows, img1.cols, CV_32FC1, h_ifft_correlation_result.data());

        cv::Point raw_peak_loc;
        double raw_max_val;
        cv::minMaxLoc(correlation_mat, nullptr, &raw_max_val, nullptr, &raw_peak_loc);
        
        cv::Mat shifted_correlation_mat = correlation_mat.clone();
        int cx = shifted_correlation_mat.cols / 2;
        int cy = shifted_correlation_mat.rows / 2;

        cv::Mat q0(shifted_correlation_mat, cv::Rect(0, 0, cx, cy));   // Top-Left
        cv::Mat q1(shifted_correlation_mat, cv::Rect(cx, 0, cx, cy));  // Top-Right
        cv::Mat q2(shifted_correlation_mat, cv::Rect(0, cy, cx, cy));  // Bottom-Left
        cv::Mat q3(shifted_correlation_mat, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

        cv::Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        cv::Point peakLoc;
        double max_val_double;
        cv::minMaxLoc(shifted_correlation_mat, nullptr, &max_val_double, nullptr, &peakLoc);

        int shift_x = peakLoc.x - cx;
        int shift_y = peakLoc.y - cy;

        // print results to terminal
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Displacement of " << pathToImg1 << " (dx, dy) is: (" << shift_x << ", " << shift_y << ") pixel" << std::endl;
        saveCorrelationResultAsImage(shifted_correlation_mat, img1.rows, img1.cols, pathToImg1);
        std::cout << "----------------------------------------" << std::endl;
        // save results to write them to disk later
        m_calculationResults.emplace_back(pathToImg1, pathToImg2, shift_x, shift_y, max_val_double);

        // Free memory on GPU
        if (nullptr != pImgGPU1)
            cudaFree(pImgGPU1);
        if (nullptr != pImgGPU2)
            cudaFree(pImgGPU2);
        if (nullptr != pCrossCorrellationMatrix)
            cudaFree(pCrossCorrellationMatrix);
        if (nullptr != pCorrellationResultReal)
            cudaFree(pCorrellationResultReal);
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
        #ifdef FIREDATASET
        if (image.rows != 2912 && image.cols != 2912)
        {
            std::cerr << "Invalid image dimension" << std::endl;
            return {};
        }

        // Define FIRE dataset ROI for cropping
        int roiWidth = 2265;
        int roiHeight = 1440;

        int startX = (image.cols - roiWidth) / 2;
        int startY = (image.rows - roiHeight) / 2;

        // sanity check
        if (startX < 0 || startY < 0 || startX + roiWidth > image.cols || startY + roiHeight > image.rows)
        {
            std::cerr << "ROI is outside of the image." << std::endl;
            return {};
        }

        cv::Rect roi(startX, startY, roiWidth, roiHeight);
        cv::Mat croppedImage;
        image(roi).copyTo(croppedImage);
        #ifdef DEBUG
        static int counter = 0;
        std::string debugPath = "./output/test_" + std::to_string(counter) + ".jpg";
        std::cout << "Saving image under path " << debugPath << std::endl;
        cv::imwrite(debugPath, croppedImage);
        counter++;
        #endif // DEBUG
        return croppedImage;
        #endif
        return image;
    }

    float *copyToGpu(const cv::Mat &imgData)
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

        // convert greyscale data to cufftComplex - we could do this on GPU in order to copy less data from CPU to GPU but for the sake on simplicity we do the conversion on CPU
        size_t num_pixels = imgData.total();

        std::vector<float> host_real_buffer(num_pixels);
        const uint8_t *img_ptr = imgData.ptr<uint8_t>(0);
        for (size_t i = 0; i < num_pixels; ++i)
        {
            host_real_buffer[i] = static_cast<float>(img_ptr[i]);
        }

        float *d_image = nullptr;
        size_t image_bytes = num_pixels * sizeof(float);
        auto retAlloc = cudaMalloc((void **)&d_image, image_bytes);
        if (cudaError::cudaSuccess != retAlloc)
        {
            std::cerr << "Unable to allocate memory on GPU" << std::endl;
            return nullptr;
        }
        auto retCpy = cudaMemcpy(d_image, host_real_buffer.data(), image_bytes, cudaMemcpyHostToDevice);
        if (cudaError::cudaSuccess != retCpy)
        {
            std::cerr << "Unable to allocate memory on GPU" << std::endl;
            cudaFree(d_image); // avoid memory leak on GPU
            return nullptr;
        }
        return d_image;
    }

    cufftComplex *calculateCrossCorrelation(cufftReal *img1, cufftReal *img2, int rows, int cols)
    {
        auto pfrequencySpaceImg1 = perform2DFFT(img1, rows, cols);
        if (nullptr == pfrequencySpaceImg1) {
            std::cerr << "Invalid pointer to pfrequencySpaceImg1 received - terminating processing" << std::endl;
            return nullptr;
        }
        auto pfrequencySpaceImg2 = perform2DFFT(img2, rows, cols);
        if (nullptr == pfrequencySpaceImg2)
        {
            cudaFree(pfrequencySpaceImg1);
            std::cerr << "Invalid pointer to pfrequencySpaceImg2 received - terminating processing" << std::endl;
            return nullptr;
        }

        // allocate memory for the cross correllation on GPU
        int complex_output_cols = cols / 2 + 1; // since we do R2C conversion and cufft does not save the redundand data (since R2C is symetric to the amplitude axis in frequency domain) of the fourrier transform
        cufftComplex *d_crossCorrelationResult = nullptr;
        size_t cross_correlation_bytes = rows * complex_output_cols * sizeof(cufftComplex);
        if (cudaError::cudaSuccess != cudaMalloc((void **)&d_crossCorrelationResult, cross_correlation_bytes)) {
            std::cerr << "Could not allocate memory for the cross correllation result" << std::endl;
            cudaFree(pfrequencySpaceImg1);
            cudaFree(pfrequencySpaceImg2);
            return nullptr;
        }

        // kernel launch parameter
        dim3 blockSize(16, 16);
        dim3 gridSize((complex_output_cols + blockSize.x - 1) / blockSize.x,
                      (rows + blockSize.y - 1) / blockSize.y);

        // Do the calculation in frequency space
        crossCorrelationKernel<<<gridSize, blockSize>>>(pfrequencySpaceImg1, pfrequencySpaceImg2, d_crossCorrelationResult, rows, complex_output_cols);
        
        // wait for all calculations to finish
        cudaDeviceSynchronize(); 
        
        cudaFree(pfrequencySpaceImg1);
        cudaFree(pfrequencySpaceImg2);
        return d_crossCorrelationResult;
    }

    cufftComplex *perform2DFFT(cufftReal *input_gpu_data, int rows, int cols)
    {
        if (!input_gpu_data) {
            std::cerr << "Received nullptr - can not do a fourrier transform" << std::endl;
            return nullptr;
        }

        // create a plan for the transformation
        long long n[2]; // 2D fourrier transform
        n[0] = rows;
        n[1] = cols;
        cufftHandle plan;
        cufftPlan2d(&plan, n[0], n[1], CUFFT_R2C);
        
        // allocate memory for the result
        int complex_output_cols = cols / 2 + 1; // since we do R2C conversion and cufft does not save the redundand data (since R2C is symetric to the amplitude axis in frequency domain) of the fourrier transform
        cufftComplex *d_fft_output = nullptr;
        size_t fft_output_bytes = rows * complex_output_cols * sizeof(cufftComplex);
        if (cudaError::cudaSuccess != cudaMalloc((void **)&d_fft_output, fft_output_bytes)) {
            std::cerr << "Could not allocate memory for the result of the fourrier transform" << std::endl;
            return nullptr;
        }

        // do the actual transformation to frquence space
        cufftExecR2C(plan, (cufftReal *)input_gpu_data, d_fft_output); // This was previously the issue - we can not hard-cast a cufftComplex_t to cufftReal_t since the frequency component will be interpreted as a real component and only the half of the data will be interpreted
        
        // wait for the fourrier transform to be fully executed on device
        cudaDeviceSynchronize();
        
        // clean up the space
        cufftDestroy(plan);
        return d_fft_output; // Gib den Zeiger auf die komplexen FFT-Ergebnisse zurück
    }

    float *perform2DIFFT_and_scale(cufftComplex *output_complex_data, int rows, int cols)
    {
        if (!output_complex_data)
        {
            std::cerr << "ERROR: Invalid data for the back transformation-received a nullptr" << std::endl;
            return nullptr;
        }

        cufftHandle plan;
        long long n[2];
        n[0] = rows;
        n[1] = cols;

        cufftPlan2d(&plan, n[0], n[1], CUFFT_C2R);
        
        // allocate memory for the back transformation of GPU - we want to have a rows x cols image as a result and since cufftComplex.x/y is a float in the background, we need floats
        float *d_ifft_output = nullptr;
        size_t ifft_output_bytes = rows * cols * sizeof(float);
        if (cudaError::cudaSuccess != cudaMalloc((void **)&d_ifft_output, ifft_output_bytes)) {
            std::cerr << "Could not allocate memory for the backtransform result" << std::endl;
            return nullptr;
        }
        
        // do the backtransformation and wait to finish them all
        cufftExecC2R(plan, output_complex_data, d_ifft_output);
        cudaDeviceSynchronize(); 
        cufftDestroy(plan);

        // we need to scale the result of cufft by the number of datapoints since we receive IFFT(FFT(x)) = N*x if we use cufft
        int num_elements = rows * cols;
        float scale_factor = static_cast<float>(rows * cols);
        dim3 scaleBlockSize(256);
        dim3 scaleGridSize((num_elements + scaleBlockSize.x - 1) / scaleBlockSize.x);
        scale_kernel<<<scaleGridSize, scaleBlockSize>>>(d_ifft_output, num_elements, scale_factor);
        cudaDeviceSynchronize(); // wait until all threads of the kernel are finished
        
        return d_ifft_output; // Gib den Zeiger auf das rekonstruierte (skalierte) Bild zurück
    }

    void saveCorrelationResultAsImage(cv::Mat &correllationMat, int rows, int cols, const std::string &originalPath)
    {
        std::string base_name = originalPath.substr(originalPath.find_last_of("/\\") + 1);
        size_t dot_pos = base_name.rfind('.');
        if (dot_pos != std::string::npos)
        {
            base_name = base_name.substr(0, dot_pos);
        }
        std::string output_filename = m_outputFolder + base_name + "_correlation_result.png";
        
        cv::Mat display_image;
        double minVal, maxVal;
        cv::minMaxLoc(correllationMat, &minVal, &maxVal);

        correllationMat.convertTo(display_image, CV_8UC1, 255.0 / maxVal);
        bool success = cv::imwrite(output_filename, display_image);
        if (success)
        {
            std::cout << "Crosscorrellation matrix successfully saved: " << output_filename << std::endl;
        }
        else
        {
            std::cerr << "Could not save crosscorrellation matrice: " << output_filename << std::endl;
        }
    }

    std::string m_outputFolder;
    int m_deviceCount{0};
    std::vector<ProcessingResult> m_calculationResults;
};