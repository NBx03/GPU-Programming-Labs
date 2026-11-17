// ЛР3. Фильтры изображений на двух GPU (Multi-GPU CUDA).
//
// Изображение разбивается на 2 горизонтальные полосы:
//   GPU 0: строки [0, h/2), с halo снизу (+r строк)
//   GPU 1: строки [h/2, h), с halo сверху (-r строк)
//
// Каждый GPU управляется своим потоком OpenMP, использует собственный контекст CUDA,
// потоки cudaStream_t и замеры через cudaEvent.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <string>
#include <vector>

#include "../common/cuda_check.h"
#include "../lab2/cpu_filters.h"
#include "../lab2/filter_common.h"
#include "../lab2/image_io.h"

__constant__ float c_w[2 * MAX_RADIUS + 1];

// Ядро размытия для полосы с учетом локального смещения halo
__global__ void blur_strip(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                           int w, int total_h, int strip_h, int r, int in_y_offset, int is_top) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= strip_h) return;

    for (int c = 0; c < 3; ++c) {
        float acc = 0.f;
        for (int dy = -r; dy <= r; ++dy) {
            // Координата внутри входного буфера полосы
            int iy = in_y_offset + y + dy;
            // Граничное условие: верхний край картинки для top, нижний край для bottom
            if (is_top && iy < 0) iy = 0;
            if (!is_top && iy >= (strip_h + r)) iy = strip_h + r - 1;

            float wy = c_w[dy + r];
            for (int dx = -r; dx <= r; ++dx) {
                int ix = clampi(x + dx, 0, w - 1);
                acc += (wy * c_w[dx + r]) * (float)in[((size_t)iy * w + ix) * 3 + c];
            }
        }
        out[((size_t)y * w + x) * 3 + c] = to_u8(acc);
    }
}

// Собель для полосы
__global__ void sobel_strip(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                            int w, int total_h, int strip_h, int in_y_offset, int is_top) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= strip_h) return;

    int l[9];
    for (int dy = -1; dy <= 1; ++dy) {
        int iy = in_y_offset + y + dy;
        if (is_top && iy < 0) iy = 0;
        if (!is_top && iy >= (strip_h + 1)) iy = strip_h;

        for (int dx = -1; dx <= 1; ++dx) {
            int ix = clampi(x + dx, 0, w - 1);
            const uint8_t* p = in + ((size_t)iy * w + ix) * 3;
            l[(dy + 1) * 3 + (dx + 1)] = luma(p[0], p[1], p[2]);
        }
    }
    uint8_t v = sobel_from_luma(l);
    uint8_t* o = out + ((size_t)y * w + x) * 3;
    o[0] = o[1] = o[2] = v;
}

// Медиана 3x3 для полосы
__global__ void median_strip(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                             int w, int total_h, int strip_h, int in_y_offset, int is_top) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= strip_h) return;

    for (int c = 0; c < 3; ++c) {
        uint8_t v[9];
        for (int dy = -1; dy <= 1; ++dy) {
            int iy = in_y_offset + y + dy;
            if (is_top && iy < 0) iy = 0;
            if (!is_top && iy >= (strip_h + 1)) iy = strip_h;

            for (int dx = -1; dx <= 1; ++dx) {
                int ix = clampi(x + dx, 0, w - 1);
                v[(dy + 1) * 3 + (dx + 1)] = in[((size_t)iy * w + ix) * 3 + c];
            }
        }
        out[((size_t)y * w + x) * 3 + c] = median9(v);
    }
}

enum class Filter { Blur, Sobel, Median };

struct GpuDevTimes {
    float h2d = 0, kernel = 0, d2h = 0;
};

// Функция обработки на 2 GPU
static void run_2gpu(Filter f, int radius, const uint8_t* h_in, uint8_t* h_out,
                     int w, int h, GpuDevTimes dev_times[2]) {
    int r = (f == Filter::Blur) ? radius : 1;
    int h0 = h / 2;
    int h1 = h - h0;

    // Подготовка весов Гаусса для обоих GPU
    float weights[2 * MAX_RADIUS + 1];
    if (f == Filter::Blur) {
        make_gauss_weights(r, weights);
    }

    #pragma omp parallel num_threads(2)
    {
        int dev = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(dev));

        if (f == Filter::Blur) {
            CUDA_CHECK(cudaMemcpyToSymbol(c_w, weights, (2 * r + 1) * sizeof(float)));
        }

        int strip_h = (dev == 0) ? h0 : h1;
        int in_y_start = (dev == 0) ? 0 : (h0 - r);
        int in_h = strip_h + r; // с halo
        int in_y_offset = (dev == 0) ? 0 : r;
        int is_top = (dev == 0) ? 1 : 0;

        size_t in_bytes = (size_t)w * in_h * 3;
        size_t out_bytes = (size_t)w * strip_h * 3;

        uint8_t *d_in = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

        EventTimer t_h2d, t_kernel, t_d2h;

        // Копирование полосы с halo Host -> Device
        const uint8_t* src_ptr = h_in + ((size_t)in_y_start * w) * 3;
        t_h2d.start();
        CUDA_CHECK(cudaMemcpy(d_in, src_ptr, in_bytes, cudaMemcpyHostToDevice));
        dev_times[dev].h2d = t_h2d.stop();

        // Запуск ядра
        dim3 block(16, 16);
        dim3 grid((w + block.x - 1) / block.x, (strip_h + block.y - 1) / block.y);

        t_kernel.start();
        if (f == Filter::Blur) {
            blur_strip<<<grid, block>>>(d_in, d_out, w, h, strip_h, r, in_y_offset, is_top);
        } else if (f == Filter::Sobel) {
            sobel_strip<<<grid, block>>>(d_in, d_out, w, h, strip_h, in_y_offset, is_top);
        } else {
            median_strip<<<grid, block>>>(d_in, d_out, w, h, strip_h, in_y_offset, is_top);
        }
        CUDA_CHECK(cudaGetLastError());
        dev_times[dev].kernel = t_kernel.stop();

        // Копирование вычисленной полосы Device -> Host
        uint8_t* dst_ptr = h_out + ((size_t)((dev == 0) ? 0 : h0) * w) * 3;
        t_d2h.start();
        CUDA_CHECK(cudaMemcpy(dst_ptr, d_out, out_bytes, cudaMemcpyDeviceToHost));
        dev_times[dev].d2h = t_d2h.stop();

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
}

int main(int argc, char** argv) {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < 2) {
        std::fprintf(stderr, "Внимание: найдено только %d GPU! Требуется 2 GPU.\n", dev_count);
        return 1;
    }

    if (argc < 4) {
        std::printf("Использование: filters_2gpu <in.png> <out.png> <blur|sobel|median> [--radius R]\n");
        return 0;
    }

    std::string in_path = argv[1];
    std::string out_path = argv[2];
    std::string f_str = argv[3];

    int radius = 3;
    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--radius" && i + 1 < argc) radius = std::atoi(argv[++i]);
    }

    Filter filter = Filter::Blur;
    if (f_str == "sobel") filter = Filter::Sobel;
    else if (f_str == "median") filter = Filter::Median;

    Image in_img;
    std::string err;
    if (!read_png(in_path, in_img, err)) {
        std::fprintf(stderr, "Ошибка чтения PNG: %s (%s)\n", in_path.c_str(), err.c_str());
        return 1;
    }
    int w = in_img.w;
    int h = in_img.h;
    Image out_img;
    out_img.w = w;
    out_img.h = h;
    out_img.rgb.resize((size_t)w * h * 3);

    std::printf("=== Multi-GPU (2 GPUs): %s (%dx%d) ===\n", in_path.c_str(), w, h);
    for (int d = 0; d < 2; ++d) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);
        std::printf("GPU %d: %s (sm_%d%d)\n", d, prop.name, prop.major, prop.minor);
    }

    GpuDevTimes times[2];
    auto t0 = std::chrono::steady_clock::now();
    run_2gpu(filter, radius, in_img.rgb.data(), out_img.rgb.data(), w, h, times);
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("\n--- Времена этапов на устройствах (мс) ---\n");
    std::printf("Device |  H2D (мс)  |  Kernel (мс)  |  D2H (мс)  | Сумма GPU\n");
    for (int d = 0; d < 2; ++d) {
        float sum = times[d].h2d + times[d].kernel + times[d].d2h;
        std::printf("GPU %d  |  %8.2f  |  %11.2f  |  %8.2f  |  %8.2f\n",
                    d, times[d].h2d, times[d].kernel, times[d].d2h, sum);
    }
    std::printf("Total 2-GPU pipeline (wall clock): %.2f ms\n", total_ms);

    // Сверка с CPU-эталоном для доказательства корректности
    std::vector<uint8_t> cpu_ref((size_t)w * h * 3);
    if (filter == Filter::Blur) cpu_blur(in_img.rgb.data(), cpu_ref.data(), w, h, radius, 4);
    else if (filter == Filter::Sobel) cpu_sobel(in_img.rgb.data(), cpu_ref.data(), w, h, 4);
    else cpu_median(in_img.rgb.data(), cpu_ref.data(), w, h, 4);

    long long diff_bytes = 0;
    int max_diff = 0;
    for (size_t i = 0; i < cpu_ref.size(); ++i) {
        int d = std::abs((int)out_img.rgb[i] - (int)cpu_ref[i]);
        if (d != 0) { diff_bytes++; max_diff = std::max(max_diff, d); }
    }
    std::printf("GPU vs CPU diff: %lld bytes / max diff %d\n", diff_bytes, max_diff);

    if (!write_png(out_path, out_img, err)) {
        std::fprintf(stderr, "Ошибка записи PNG: %s\n", err.c_str());
        return 1;
    }
    std::printf("Результат сохранен в %s\n", out_path.c_str());
    return 0;
}
