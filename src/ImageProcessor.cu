#include "cudaAtScaleFinalAssignment/ImageProcessor.hpp"

__global__ void crossCorrelationKernel(cufftComplex* F1_gpu, cufftComplex* F2_gpu, cufftComplex* P_gpu,
                                       int rows, int complex_output_cols)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x; // define u by x component of the block and thread idx
    int v = blockIdx.y * blockDim.y + threadIdx.y; // define v by y component of the block and thread idx

    // check if u and v (reduced size since the R2C transformation has only half the size of the original image) is within the image pixel coordinates
    if (v < rows && u < complex_output_cols)
    {
        // determine the index in the continous memory
        int idx = v * complex_output_cols + u;
        
        // get fourrier transformed values
        cufftComplex F1 = F1_gpu[idx];
        cufftComplex F2 = F2_gpu[idx];

        // calculate complex conjugate - F2_complKonj.x = F2.x, F2_complKonj.y = -F2.y
        cufftComplex F2_conj;
        F2_conj.x = F2.x;
        F2_conj.y = -F2.y;

        // calculate numerator G = F1 * F2_complKonj
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        // here: (F1.x + F1.y*i) * (F2_conj.x + F2_conj.y*i)
        // so (F1.x + F1.y*i) * (F2.x - F2.y*i)
        cufftComplex G;
        G.x = F1.x * F2_conj.x - F1.y * F2_conj.y; // Re
        G.y = F1.x * F2_conj.y + F1.y * F2_conj.x; // Im

        // calculate abs(G) = sqrt(G.x^2 + G.y^2)
        float abs_G = sqrtf(G.x * G.x + G.y * G.y);

        // calculate P(u,v) = G / abs(G)
        if (abs_G > 1e-9f) // avoid division by (nearly) zero
        {
            P_gpu[idx].x = G.x / abs_G;
            P_gpu[idx].y = G.y / abs_G;
        }
        else // in case of division by zero - set values hardly to 0
        {
            P_gpu[idx].x = 0.0f;
            P_gpu[idx].y = 0.0f;
        }
    }
}