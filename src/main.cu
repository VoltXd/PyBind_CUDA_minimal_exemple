#include <pybind11/pybind11.h> 
#include <pybind11/numpy.h> 
#include <vector>
#include <random>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include <iostream>

constexpr size_t CUDA_THREADS_PER_BLOCK = 1024;
constexpr size_t CUDA_BLOCKS = 32;

constexpr size_t CUDA_THREADS_COUNT = CUDA_BLOCKS * CUDA_THREADS_PER_BLOCK;

namespace py = pybind11;

double add(double a, double b)
{
    return a + b;
}

__global__ void addVector(const double* a, const double* b, double* c, uint64_t N)
{
    const uint64_t startIdx = threadIdx.x + blockDim.x * blockIdx.x;
    const uint64_t step = blockDim.x * gridDim.x;

    for (uint64_t i = startIdx; i < N; i += step)
        c[i] = a[i] + b[i];
}

py::array_t<double> CUDA_addVector(py::array_t<double>& a, py::array_t<double>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Input shapes must match!");

    double* cuPtr_a;
    double* cuPtr_b;
    double* cuPtr_c;

    cudaMalloc(&cuPtr_a, a.size() * sizeof(double));
    cudaMalloc(&cuPtr_b, a.size() * sizeof(double));
    cudaMalloc(&cuPtr_c, a.size() * sizeof(double));

    cudaMemcpy(cuPtr_a, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuPtr_b, b.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice);

    addVector<<<CUDA_BLOCKS,CUDA_THREADS_PER_BLOCK>>>(cuPtr_a, cuPtr_b, cuPtr_c, a.size());

    std::vector<double> c(a.size());
    cudaMemcpy(c.data(), cuPtr_c, a.size() * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "[" << c[0] << ' ' << c[a.size() - 1] << "]\n";

    return py::array(c.size(), c.data());
}

py::array_t<double> getRandomGaussVector(double mean, double stddev, uint64_t N)
{
    std::random_device r;
    std::default_random_engine generator { r() };
    std::normal_distribution gaussianDistribution(mean, stddev);

    std::vector<double> randomNumbers(N);
    for (double& d : randomNumbers)
        d = gaussianDistribution(generator);

    return py::array(randomNumbers.size(), randomNumbers.data());
}

__global__ void kernel_curandSetup(curandState* globalState, uint64_t seed)
{
    size_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, threadIndex, 0, &globalState[threadIndex]);
}

__global__ void kernel_getRandomGaussVector(curandState* globalState, double* v, double mean, double stddev, uint64_t N)
{
    const size_t startIndex = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t step = blockDim.x * gridDim.x;
    
    curandState localState = globalState[startIndex];

    
    for (uint64_t idx = startIndex; idx < N; idx += step)
        v[idx] = stddev * curand_normal(&localState) + mean;

    
    globalState[startIndex] = localState;
}

py::array_t<double> getRandomGaussVector_cuda(double mean, double stddev, uint64_t N)
{
    // Random device allocation
    curandState* cuPtr_deviceCurandStates;
    cudaMalloc(&cuPtr_deviceCurandStates, CUDA_THREADS_COUNT * sizeof(curandState));
    kernel_curandSetup<<<CUDA_BLOCKS,CUDA_THREADS_PER_BLOCK>>>(cuPtr_deviceCurandStates, time(NULL));

    double* cuPtr_randomNumbers;
    cudaMalloc(&cuPtr_randomNumbers, N * sizeof(double));

    kernel_getRandomGaussVector<<<CUDA_BLOCKS,CUDA_THREADS_PER_BLOCK>>>(cuPtr_deviceCurandStates, cuPtr_randomNumbers, mean, stddev, N);

    std::vector<double> randomNumbers(N);
    cudaMemcpy(randomNumbers.data(), cuPtr_randomNumbers, N * sizeof(double), cudaMemcpyDeviceToHost);

    return py::array(randomNumbers.size(), randomNumbers.data());
}


PYBIND11_MODULE(TEST_PYBIND_CUDA, handle)
{
    handle.doc() = "Docu";
    handle.def("add", &add);
    handle.def("add_cuda", &CUDA_addVector);
    handle.def("get_random_gauss_vector", &getRandomGaussVector);
    handle.def("get_random_gauss_vector_cuda", &getRandomGaussVector_cuda);
}