#include <cuda_runtime.h>
#include <torch/extension.h>


namespace {

torch::Tensor wrap_cuda_malloc_host(size_t size) {
    void *p;
    cudaError_t err = cudaMallocHost(&p, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << (int)err << ": " << cudaGetErrorString(err) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        abort();
    }
    auto deleter = [](void *p) {
        cudaError_t err = cudaFreeHost(p);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error " << (int)err << ": " << cudaGetErrorString(err) << " " << __FILE__ << ":" << __LINE__ << std::endl;
            abort();
        }
    };
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    return torch::from_blob(p, {(int64_t)size}, {1}, deleter, options);
}

void wrap_cuda_memcpy_2d_async(intptr_t dst_intptr, size_t dpitch, intptr_t src_intptr, size_t spitch, size_t width, size_t height, int cuda_memcpy_kind, intptr_t stream_intptr) {
    void *dst = reinterpret_cast<void *>(dst_intptr);
    void const *src = reinterpret_cast<void const *>(src_intptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_intptr);
    cudaError_t err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, (cudaMemcpyKind)cuda_memcpy_kind, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << (int)err << ": " << cudaGetErrorString(err) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        throw std::runtime_error("CUDA error " + std::to_string(err) + ": " + cudaGetErrorString(err));
    }
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wrap_cuda_memcpy_2d_async", &wrap_cuda_memcpy_2d_async);
    m.def("wrap_cuda_malloc_host", &wrap_cuda_malloc_host);
}
