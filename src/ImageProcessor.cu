#include "cudaAdvancedLibsFinalAssignment/ImageProcessor.hpp"

__global__ void crossCorrelationKernel(cufftComplex* F1_gpu, cufftComplex* F2_gpu, cufftComplex* P_gpu,
                                       int rows, int complex_output_cols)
{
    // determine image coordinates based on thread id for the processing
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    // sanity check if the determined thread/image coordinates exist in the image
    if (v < rows && u < complex_output_cols)
    {
        int idx = v * complex_output_cols + u; // roll out the indices to the array representation of the image
        
        cufftComplex F1 = F1_gpu[idx];
        cufftComplex F2 = F2_gpu[idx];

        // calculate complex conjugate conj(a+bi) = a-bi
        cufftComplex F2_conj;
        F2_conj.x = F2.x;
        F2_conj.y = -F2.y;

        // calculate numerator: G = F1 * F2_conj
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        // (F1.x + F1.y*i) * (F2_conj.x + F2_conj.y*i)
        cufftComplex G;
        G.x = F1.x * F2_conj.x - F1.y * F2_conj.y; // Re
        G.y = F1.x * F2_conj.y + F1.y * F2_conj.x; // Im

        // caculate |G| = sqrt(G.x^2 + G.y^2)
        float abs_G = sqrtf(G.x * G.x + G.y * G.y);

        // P(u,v) = G / |G|
        // Avoid and approximate division by zero
        if (abs_G > 1e-9f)
        {
            P_gpu[idx].x = G.x / abs_G;
            P_gpu[idx].y = G.y / abs_G;
        }
        else
        {
            P_gpu[idx].x = 0.0f;
            P_gpu[idx].y = 0.0f;
        }
    }
}

__global__ void scale_kernel(float* data, int num_elements, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] /= scale_factor;
    }
}