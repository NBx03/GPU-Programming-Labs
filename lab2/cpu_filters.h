#pragma once
#include <cstdint>

// Все изображения: RGB, 8 бит на канал, чересстрочно (w*h*3 байт). Границы: clamp координат.
// threads = число потоков OpenMP (1 -- однопоточный эталон).
void cpu_blur(const uint8_t* in, uint8_t* out, int w, int h, int radius, int threads);
void cpu_sobel(const uint8_t* in, uint8_t* out, int w, int h, int threads);
void cpu_median(const uint8_t* in, uint8_t* out, int w, int h, int threads);
int cpu_max_threads();
