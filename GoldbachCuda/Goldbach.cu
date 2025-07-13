#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <thread>

__global__ void sieveOfEratosthenes(int* outputArray, unsigned long long N) {
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

__global__ void goldbach(
    int* primesArray, 
    int* primesToIndex, 
    int* intermediateArray, 
    int* outputArray, 
    unsigned long long numPrimes,
    unsigned long long N
) {
    // Calculate the global index for the current thread
    unsigned long long cur_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_threads = gridDim.x * blockDim.x;

    unsigned long long start_index = cur_index * numPrimes;

    for (; cur_index < N/2; cur_index += num_threads) {
        unsigned long long cur_num = (cur_index + 2) * 2;
        unsigned long long numSatisfied = 0;

        for (unsigned long long i = 0; i < numPrimes; ++i) {
            int value = cur_num - primesArray[i];
            if (value < 0) break;
            if (primesToIndex[value] != -1) {
                intermediateArray[start_index + numSatisfied] = primesToIndex[value] + i + 2;
                numSatisfied += 1;
            }
        }

        outputArray[cur_index] = intermediateArray[start_index + numSatisfied / 2];
    }
}

int main() {
    unsigned long long N = 1000000; // Number of elements in the array
    unsigned long long max_primes = N / (std::log(N)-4); // Number of primes

    // 1. Allocate host memory for the output array
    int* h_outputArray = new int[N];

    // 2. Allocate device memory for the output array
    int* d_outputArray;
    cudaError_t err = cudaMalloc(&d_outputArray, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (malloc output): %s\n", cudaGetErrorString(err));
    }

    // 3. Set device memory to 1 (all prime)
    err = cudaMemset(&d_outputArray, 0, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (memset output): %s\n", cudaGetErrorString(err));
    }

    // 4. Define grid and block dimensions
    int threadsPerBlock = 512;
    int numBlocks = 10;

    // 6. Launch the kernel
    sieveOfEratosthenes << <numBlocks, threadsPerBlock >> > (d_outputArray, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (kernel): %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    printf("Successfully finished GPU Job\n");

    // 6. Copy the result back from device to host
    err = cudaMemcpy(h_outputArray, d_outputArray, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (memCpy): %s\n", cudaGetErrorString(err));
    }

    // 8. Print a few results to verify
    for (unsigned long long i = 0; i < 10; ++i) {
        std::cout << "h_outputArray[" << i << "] = " << h_outputArray[i] << std::endl;
    }
    std::cout << "..." << std::endl;
    for (unsigned long long i = N - 10; i < N; ++i) {
        std::cout << "h_outputArray[" << i << "] = " << h_outputArray[i] << std::endl;
    }

    // 9. Write to text file
    FILE* outputFile = fopen("primes.txt", "w"); // "w" for write mode
    if (outputFile != NULL) {
        for (unsigned long long i = 2; i < N; ++i) {
            if (!h_outputArray[i]) { // check prime
                fprintf(outputFile, "%d\n", i); // Write each number followed by a newline
            }
        }
    } else {
        fprintf(stderr, "Error opening primes.txt for writing.\n");
    }

    // 10. Create primes arrays
    int* h_primesArray = new int[max_primes];
    int* h_primesToIndex = new int[N];
    unsigned long long primesIdx = 0;
    for (unsigned long long i = 0; i < N; ++i) {
        if (i < 2 || h_outputArray[i]) {
            h_primesToIndex[i] = -1;
        } else {
            h_primesArray[primesIdx] = i;
            h_primesToIndex[i] = primesIdx;
            ++primesIdx;
        }
    }

    // 5. Define the arrays for goldbach values
    int* d_primesArray;
    int* d_primesToIndex;
    int* d_intermediate;
    err = cudaMalloc(&d_primesArray, primesIdx * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (malloc primes): %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc(&d_primesToIndex, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (malloc primes): %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc(&d_intermediate, primesIdx * threadsPerBlock * numBlocks * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (malloc primes): %s\n", cudaGetErrorString(err));
    }
    err = cudaMemset(&d_intermediate, 0, primesIdx * threadsPerBlock * numBlocks * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (memset primes): %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_primesArray, h_primesArray, primesIdx * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_primesToIndex, h_primesToIndex, N * sizeof(int), cudaMemcpyHostToDevice);

    // 11. Goldbach
    goldbach << <numBlocks, threadsPerBlock >> > (
        d_primesArray,
        d_primesToIndex,
        d_intermediate,
        d_outputArray,
        primesIdx,
        N
    );

    // 6. Copy the result back from device to host
    err = cudaMemcpy(h_outputArray, d_outputArray, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (memCpy): %s\n", cudaGetErrorString(err));
    }

    // 8. Print a few results to verify
    for (unsigned long long i = 0; i < 10; ++i) {
        std::cout << "h_outputArray[" << i << "] = " << h_outputArray[i] << std::endl;
    }
    std::cout << "..." << std::endl;
    for (unsigned long long i = N/2 - 10; i < N/2; ++i) {
        std::cout << "h_outputArray[" << i << "] = " << h_outputArray[i] << std::endl;
    }

    // 9. Write to text file
    outputFile = fopen("goldbach.txt", "w"); // "w" for write mode
    if (outputFile != NULL) {
        for (unsigned long long i = 0; i < N/2; ++i) {
            fprintf(outputFile, "%d %d\n", (i+2)*2, h_outputArray[i]); // Write each number followed by a newline
        }
    }
    else {
        fprintf(stderr, "Error opening goldbach.txt for writing.\n");
    }


    // 11. Free device memory
    cudaFree(d_primesArray);
    cudaFree(d_primesToIndex);
    cudaFree(d_intermediate);
    cudaFree(d_outputArray);

    // 12. Free host memory
    delete[] h_outputArray;
    delete[] h_primesArray;
    delete[] h_primesToIndex;

    return 0;
}