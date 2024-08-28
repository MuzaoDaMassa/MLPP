#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAddGPU(const double *A, const double *B, double *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

// Kernel function to perform matrix multiplication
__global__ void matMulGPU(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        double value = 0.0;
        for (int e = 0; e < N; ++e) {
            value += A[row * N + e] * B[e * K + col];
        }
        C[row * K + col] = value;
    }
}

extern "C" void launch_vector_add(double* A, double* B, double* C, int numElements) 
{
    //int numElements = A.size();
    size_t size = numElements * sizeof(double);

    double *h_A = A;
    double *h_B = B;
    double *h_C = C;

    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (double)RAND_MAX;
        h_B[i] = rand() / (double)RAND_MAX;
    }

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            std::cerr << "Result verification failed at element " << i << std::endl;
            break;
        }
    }

    std::cout << "Vector addition completed successfully." << std::endl;
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

extern "C" void launch_mat_mul(double* A, double* B, double* C, int M, int N, int K) 
{
    // Calculate the size of the matrices
    size_t size_A = M * N * sizeof(double);
    //size_t size_B = N * K * sizeof(double);
    //size_t size_C = M * K * sizeof(double);

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_A);
    cudaMalloc((void**)&d_C, size_A);

    // Copy input data from host to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_A, cudaMemcpyHostToDevice);

    // Determine the block and grid dimensions
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the matrix multiplication kernel
    matMulGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy the result from device to host
    cudaMemcpy(C, d_C, size_A, cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Matrix multiplication completed successfully." << std::endl;
}

