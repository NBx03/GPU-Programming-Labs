#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err_), #call);              \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

// Замер интервала на устройстве через cudaEvent (поток 0). Возвращает миллисекунды.
struct EventTimer {
    cudaEvent_t a_{}, b_{};
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&a_));
        CUDA_CHECK(cudaEventCreate(&b_));
    }
    ~EventTimer() {
        cudaEventDestroy(a_);
        cudaEventDestroy(b_);
    }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    void start() { CUDA_CHECK(cudaEventRecord(a_, 0)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(b_, 0));
        CUDA_CHECK(cudaEventSynchronize(b_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a_, b_));
        return ms;
    }
};
