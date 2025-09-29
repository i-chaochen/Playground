// hipcc -std=c++17 sync_test.cpp -o sync_test -lpthread
#include <hip/hip_runtime.h>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstdlib>

#define HIP_CHECK(cmd)                                                   \
    do {                                                                 \
        hipError_t e = cmd;                                              \
        if (e != hipSuccess) {                                           \
            std::cerr << "HIP error: " << hipGetErrorString(e)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";  \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)

// Dummy kernel: just increments values
__global__ void testKernel(float* data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += 1.0f;
}

struct PerDeviceData {
    int deviceId;
    hipDeviceProp_t props;
    std::vector<hipStream_t> streams;
    std::vector<void*> d_inputs;
    std::vector<void*> d_outputs;
    std::vector<void*> h_pinned_src;
    std::vector<void*> h_pinned_dst;
    size_t bufBytes = 0;
};

int main() {
    // === Thread pool to claim all CPU cores ===
    unsigned nThreads = std::thread::hardware_concurrency();
    std::atomic<bool> keepRunning{true};
    std::vector<std::thread> cpuThreads;
    cpuThreads.reserve(nThreads);

    for (unsigned i = 0; i < nThreads; i++) {
        cpuThreads.emplace_back([&keepRunning]() {
            std::vector<char> buffer(1024 * 1024, 42); // 1MB per thread
            while (keepRunning) {
                // Simulate host-only copy
                std::vector<char> temp(buffer);
                temp[0] = buffer[0];
            }
        });
    }

    // === Query devices ===
    int nDevices = 0, usedDevices = 4;
    HIP_CHECK(hipGetDeviceCount(&nDevices));

    std::cout << "Found " << nDevices << " GPU devices, but we only used " << usedDevices << " GPUs\n";

    int streamsPerDevice = 4;
    std::vector<PerDeviceData> devices(streamsPerDevice);

    for (int d = 0; d < usedDevices; d++) {
        HIP_CHECK(hipSetDevice(d));
        devices[d].deviceId = d;

        HIP_CHECK(hipGetDeviceProperties(&devices[d].props, d));
        std::cout << "GPU " << d << ": " << devices[d].props.name << "\n";

        // Query free memory
        size_t freeMem = 0, totalMem = 0;
        HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));
        freeMem = freeMem * 0.4;

        size_t allocSize = freeMem > (64ULL << 20) ? freeMem - (64ULL << 20) : freeMem / 2;
        devices[d].bufBytes = allocSize / (2 * streamsPerDevice);

        std::cout << "  Total=" << (totalMem >> 20) << " MB, Free="
                  << (freeMem >> 20) << " MB, Allocating ~"
                  << (allocSize >> 20) << " MB across " << streamsPerDevice
                  << " streams\n";

        // Allocate per stream
        devices[d].streams.resize(streamsPerDevice);
        devices[d].d_inputs.resize(streamsPerDevice);
        devices[d].d_outputs.resize(streamsPerDevice);
        devices[d].h_pinned_src.resize(streamsPerDevice);
        devices[d].h_pinned_dst.resize(streamsPerDevice);

        for (int s = 0; s < streamsPerDevice; ++s) {
            HIP_CHECK(hipStreamCreateWithFlags(&devices[d].streams[s], hipStreamNonBlocking));
            HIP_CHECK(hipMalloc(&devices[d].d_inputs[s], devices[d].bufBytes));
            HIP_CHECK(hipMalloc(&devices[d].d_outputs[s], devices[d].bufBytes));
            HIP_CHECK(hipHostMalloc(&devices[d].h_pinned_src[s], devices[d].bufBytes, hipHostMallocDefault));
            HIP_CHECK(hipHostMalloc(&devices[d].h_pinned_dst[s], devices[d].bufBytes, hipHostMallocDefault));

            // Fill pinned buffer
            float* hsrc = static_cast<float*>(devices[d].h_pinned_src[s]);
            size_t numFloats = devices[d].bufBytes / sizeof(float);
            for (size_t i = 0; i < numFloats; ++i) hsrc[i] = static_cast<float>(i % 1024);
        }
    }

    // === Main infinite loop ===
    size_t iter = 0;
    while (true) {
        for (int d = 0; d < usedDevices; d++) {
            HIP_CHECK(hipSetDevice(devices[d].deviceId));

            for (size_t s = 0; s < devices[d].streams.size(); ++s) {
                size_t numFloats = devices[d].bufBytes / sizeof(float);

                HIP_CHECK(hipMemcpyHtoDAsync(devices[d].d_inputs[s],
                                             devices[d].h_pinned_src[s],
                                             devices[d].bufBytes,
                                             devices[d].streams[s]));

                dim3 block(256);
                dim3 grid((numFloats + block.x - 1) / block.x);
                hipLaunchKernelGGL(testKernel, grid, block, 0, devices[d].streams[s],
                                   static_cast<float*>(devices[d].d_inputs[s]), numFloats);

                HIP_CHECK(hipMemcpyDtoHAsync(devices[d].h_pinned_dst[s],
                                             devices[d].d_inputs[s],
                                             devices[d].bufBytes,
                                             devices[d].streams[s]));
            }
        }

        // Wait for all GPUs to finish
        for (int d = 0; d < usedDevices; d++) {
            HIP_CHECK(hipSetDevice(devices[d].deviceId));
            HIP_CHECK(hipDeviceSynchronize());
        }

        std::cout << "Iteration " << iter++
                  << " done, devices=" << devices.size()
                  << ", StreamsPerDevice=" << streamsPerDevice
                  << ", hipMemcpyHtoDAsync() + hipLaunchKernelGGL() + hipMemcpyDtoHAsync() + hipDeviceSynchronize()" 
                  << std::endl;
    }

    // Cleanup (unreachable in this infinite loop)
    keepRunning = false;
    for (auto& t : cpuThreads) t.join();
    return 0;
}
