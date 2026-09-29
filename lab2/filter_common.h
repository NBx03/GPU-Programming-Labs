#pragma once
// Код, общий для CPU-эталона и CUDA-ядер: гарантирует одинаковую арифметику пикселя.
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

constexpr int MAX_RADIUS = 15;  // размер ядра размытия до 31x31

HD inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// float -> uint8 с округлением и насыщением
HD inline uint8_t to_u8(float v) {
    int x = (int)(v + 0.5f);
    return (uint8_t)clampi(x, 0, 255);
}

// целочисленная яркость (веса 77/150/29 из 256 ~ 0.299/0.587/0.114)
HD inline int luma(int r, int g, int b) { return (77 * r + 150 * g + 29 * b + 128) >> 8; }

// Оператор Собеля по яркостям окна 3x3 (row-major, l[4] -- центр)
HD inline uint8_t sobel_from_luma(const int* l) {
    int gx = -l[0] + l[2] - 2 * l[3] + 2 * l[5] - l[6] + l[8];
    int gy = -l[0] - 2 * l[1] - l[2] + l[6] + 2 * l[7] + l[8];
    float m = sqrtf((float)(gx * gx + gy * gy));
    int v = (int)(m + 0.5f);
    return (uint8_t)(v > 255 ? 255 : v);
}

#define PIX_SORT(a, b)         \
    {                          \
        if (p[a] > p[b]) {     \
            uint8_t t_ = p[a]; \
            p[a] = p[b];       \
            p[b] = t_;         \
        }                      \
    }
// Медиана 9 значений (сортирующая сеть Paeth/Devillard, 19 обменов)
HD inline uint8_t median9(uint8_t* p) {
    PIX_SORT(1, 2) PIX_SORT(4, 5) PIX_SORT(7, 8) PIX_SORT(0, 1) PIX_SORT(3, 4) PIX_SORT(6, 7)
    PIX_SORT(1, 2) PIX_SORT(4, 5) PIX_SORT(7, 8) PIX_SORT(0, 3) PIX_SORT(5, 8) PIX_SORT(4, 7)
    PIX_SORT(3, 6) PIX_SORT(1, 4) PIX_SORT(2, 5) PIX_SORT(4, 7) PIX_SORT(4, 2) PIX_SORT(6, 4)
    PIX_SORT(4, 2)
    return p[4];
}
#undef PIX_SORT

// Нормированные 1D-веса Гаусса для радиуса r (размер 2r+1); sigma как в OpenCV по размеру ядра.
inline void make_gauss_weights(int r, float* w) {
    double sigma = 0.3 * ((2 * r + 1) * 0.5 - 1) + 0.8;
    double t[2 * MAX_RADIUS + 1], sum = 0;
    for (int k = -r; k <= r; ++k) {
        t[k + r] = std::exp(-(double)(k * k) / (2 * sigma * sigma));
        sum += t[k + r];
    }
    for (int k = 0; k < 2 * r + 1; ++k) w[k] = (float)(t[k] / sum);
}
