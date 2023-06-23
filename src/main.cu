#include <pybind11/pybind11.h> 
#include <pybind11/numpy.h> 
#include <vector>

#include <iostream>

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

    addVector<<<32,1024>>>(cuPtr_a, cuPtr_b, cuPtr_c, a.size());

    std::vector<double> c(a.size());
    cudaMemcpy(c.data(), cuPtr_c, a.size() * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "[" << c[0] << ' ' << c[a.size() - 1] << "]\n";

    return py::array(c.size(), c.data());
}

PYBIND11_MODULE(TEST_PYBIND_CUDA, handle)
{
    handle.doc() = "Docu";
    handle.def("add", &add);
    handle.def("add_cuda", &CUDA_addVector);
}