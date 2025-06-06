#pragma once

#include <iostream>
#include <string>

#include <cufft.h>

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

// forward declaration of the calculation kernel
__global__ void crossCorrelationKernel(cufftComplex *F1_gpu, cufftComplex *F2_gpu, cufftComplex *P_gpu, int rows, int complex_output_cols);
__global__ void scale_kernel(float *data, int num_elements, float scale_factor);

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

    bool processImage(const std::string &pathToImg1, const std::string &pathToImg2)
    {
        std::cout << "Processing paths: " << pathToImg1 << "; " << pathToImg2 << std::endl;
        // read image to RAM
        cv::Mat img1 = readImage(pathToImg1);
        cv::Mat img2 = readImage(pathToImg2);

        if ((img1.rows != img2.rows) || (img1.cols != img2.cols)) {
            std::cerr << "Invalid image dimension" << std::endl;
            return false;
        }

        // Copy image data to GPU
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
        cudaMemcpy(h_ifft_correlation_result.data(), pCorrellationResultReal, img1.rows * img1.cols * sizeof(float), cudaMemcpyDeviceToHost);

        saveCorrelationResultAsImage(h_ifft_correlation_result, img1.rows, img1.cols, pathToImg1);
        
        // Search for maximum
        float max_val = 0.0f;
        int max_idx = 0;
        std::cout << "Image is: " << img1.rows << " x " << img1.cols << " = " << h_ifft_correlation_result.size() << std::endl;
        for (int i = 0; i < h_ifft_correlation_result.size(); ++i)
        {
            if (h_ifft_correlation_result[i] > max_val)
            {
                std::cout << "Update maxval" << std::endl;
                max_val = h_ifft_correlation_result[i];
                max_idx = i;
            }
            else if (h_ifft_correlation_result[i] == 0) {
                std::cerr << "Value is 0 " << std::endl;
            }
        }
        std::cout << "max_idx = " << max_idx << std::endl;
        std::cout << "max_val = " << max_val << std::endl;

        int peak_y = max_idx / img1.cols;
        int peak_x = max_idx % img1.cols;

        std::cout << "Crosscorrelationspeak found at (X,Y): (" << peak_x << ", " << peak_y << ") with value: " << max_val << std::endl;

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

        // convert greyscale data to cufftComplex - we could do this on GPU in order to copy less data from CPU to GPU but for the sake on simplicity we do the conversion on CPU
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

        // copy data to GPU
        err = cudaMemcpy(d_complex_image, host_complex_buffer.data(), complex_image_bytes, cudaMemcpyHostToDevice);
        if (cudaError::cudaSuccess != err)
        {
            std::cerr << "Unable to allocate memory on GPU" << std::endl;
            cudaFree(d_complex_image); // avoid memory leak on GPU
            return nullptr;
        }
        return d_complex_image;
    }

    cufftComplex * calculateCrossCorrelation(cufftComplex *img1, cufftComplex *img2, int rows, int cols)
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

    cufftComplex *perform2DFFT(cufftComplex *input_gpu_data, int rows, int cols)
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
        cufftExecR2C(plan, (cufftReal *)input_gpu_data, d_fft_output);
        
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

    void saveCorrelationResultAsImage(const std::vector<float> &correlation_data, int rows, int cols, const std::string &original_path)
    {
        cv::Mat correlation_image(rows, cols, CV_32FC1, const_cast<float *>(correlation_data.data()));
        cv::Mat display_image;
        double minVal, maxVal;
        cv::minMaxLoc(correlation_image, &minVal, &maxVal);

        std::cout << "Korrelationsbild Min-Wert: " << minVal << ", Max-Wert: " << maxVal << std::endl;

        if (maxVal > 0)
        {
            correlation_image.convertTo(display_image, CV_8UC1, 255.0 / maxVal);
        }
        else
        {
            display_image = cv::Mat::zeros(rows, cols, CV_8UC1);
            std::cerr << "WARNUNG: Max-Wert im Korrelationsbild ist 0 oder negativ. Bild wird schwarz gespeichert." << std::endl;
        }

        std::string base_name = original_path.substr(original_path.find_last_of("/\\") + 1);
        size_t dot_pos = base_name.rfind('.');
        if (dot_pos != std::string::npos)
        {
            base_name = base_name.substr(0, dot_pos);
        }
        std::string output_filename = m_outputFolder + "/" + base_name + "_correlation_result.png";

        bool success = cv::imwrite(output_filename, display_image);
        if (success)
        {
            std::cout << "Successfully saved correllation image: " << output_filename << std::endl;
        }
        else
        {
            std::cerr << "ERROR: Could not save correllation image: " << output_filename << std::endl;
        }
    }

    std::string m_outputFolder;
    int m_deviceCount{0};
};