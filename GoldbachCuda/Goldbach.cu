#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

__global__ void sieveOfEratosthenes(int* outputArray, long long N) {
    // Calculate the global index for the current thread
    unsigned long long cur_num = blockIdx.x * blockDim.x + threadIdx.x + 2;
    unsigned long long max_index = (gridDim.x - 1) * blockDim.x + (blockDim.x - 1) + 2;
    unsigned long long divisor = N / max_index;
    unsigned long long multiplier = divisor != 0 ? N / divisor : 0;
    

    for (; cur_num < N; cur_num += multiplier) {
        // Sieve of Eratosthenes for current index
        unsigned long long idx = cur_num * cur_num;
        for (; idx < N; idx += cur_num) {
            // IMPORTANT: 0 is prime, 1 is composite
            atomicExch(&outputArray[idx], 1);
        }
        if (!multiplier) break;
    }
}

int main() {
    unsigned long long N = 100000000; // Number of elements in the array

    // 1. Allocate host memory for the output array
    int* h_outputArray = new int[N];

    // 2. Allocate device memory for the output array
    int* d_outputArray;
    cudaError_t err = cudaMalloc(&d_outputArray, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (malloc): %s\n", cudaGetErrorString(err));
    }

    // 3. Set device memory to 1 (all prime)
    err = cudaMemset(&d_outputArray, 0, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (memset): %s\n", cudaGetErrorString(err));
    }

    // 4. Define grid and block dimensions
    int threadsPerBlock = 512;
    int numBlocks = 10;

    // 5. Launch the kernel
    sieveOfEratosthenes << <numBlocks, threadsPerBlock >> > (d_outputArray, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (kernel): %s\n", cudaGetErrorString(err));
    }
    printf("Successfully finished GPU Job");

    // 6. Copy the result back from device to host
    err = cudaMemcpy(h_outputArray, d_outputArray, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (memCpy): %s\n", cudaGetErrorString(err));
    }

    // 7. Free device memory
    cudaFree(d_outputArray);

    // 8. Print a few results to verify
    for (unsigned long long i = 0; i < 10; ++i) {
        std::cout << "h_outputArray[" << i << "] = " << h_outputArray[i] << std::endl;
    }
    std::cout << "..." << std::endl;
    for (unsigned long long i = N - 10; i < N; ++i) {
        std::cout << "h_outputArray[" << i << "] = " << h_outputArray[i] << std::endl;
    }

    // 9. Write to text file
    FILE* outputFile = fopen("output_cuda_c.txt", "w"); // "w" for write mode
    if (outputFile != NULL) {
        for (unsigned long long i = 2; i < N; ++i) {
            if (!h_outputArray[i]) { // check prime
                fprintf(outputFile, "%d\n", i); // Write each number followed by a newline
            }
        }
    } else {
        fprintf(stderr, "Error opening output_cuda_c.txt for writing.\n");
    }

    // 10. Free host memory
    delete[] h_outputArray;

    return 0;
}