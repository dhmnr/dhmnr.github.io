#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdio.h>

// Utility function to check CUDA errors
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Verify results
bool verifyResults(const float *a, const float *b, const float *c, int n,
                   float tolerance = 1e-5) {
  for (int i = 0; i < n; i++) {
    float expected = a[i] + b[i];
    if (fabs(c[i] - expected) > tolerance) {
      std::cerr << "Mismatch at index " << i << ": expected " << expected
                << ", got " << c[i] << std::endl;
      return false;
    }
  }
  return true;
}

// CUDA kernel for vector addition
__global__ void vectorAddKernel(const float *a, const float *b, float *c,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-stride loop for handling arrays larger than grid size
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    c[i] = a[i] + b[i];
  }
}

void vectorAdd(const float *h_a, const float *h_b, float *h_c, int n) {
  float *d_a, *d_b, *d_c;
  size_t bytes = n * sizeof(float);

  // Allocate device memory
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data from host to device
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  // Launch configuration
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  // Limit grid size for efficiency
  gridSize = (gridSize > 65535) ? 65535 : gridSize;

  // Launch kernel
  vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  // Wait for kernel to complete
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

int main(int argc, char **argv) {

  int n = 1 << 24; // 16M elements

  size_t bytes = n * sizeof(float);

  // Allocate host memory
  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<float> h_c(n);

  // Initialize random number generator
  std::random_device rd;
  std::mt19937 gen(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  // Initialize input vectors
  std::cout << "Initializing vectors..." << std::endl;
  for (int i = 0; i < n; i++) {
    h_a[i] = dist(gen);
    h_b[i] = dist(gen);
  }

  // Timed run
  std::cout << "\nPerforming timed run..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  vectorAdd(h_a.data(), h_b.data(), h_c.data(), n);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;

  std::cout << "Time: " << duration.count() << " ms" << std::endl;

  // Verify results
  std::cout << "\nVerifying results..." << std::endl;
  if (verifyResults(h_a.data(), h_b.data(), h_c.data(), n)) {
    std::cout << "Results are correct!" << std::endl;
  } else {
    std::cerr << "Results are incorrect!" << std::endl;
    return 1;
  }

  return 0;
}
