#include "cpu_filters.h"

#include <algorithm>

#include "filter_common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int cpu_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void cpu_blur(const uint8_t* in, uint8_t* out, int w, int h, int r, int threads) {
    float wt[2 * MAX_RADIUS + 1];
    make_gauss_weights(r, wt);
    const int nt = std::max(1, threads);
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                float acc = 0.f;
                for (int dy = -r; dy <= r; ++dy) {
                    int yy = clampi(y + dy, 0, h - 1);
                    float wy = wt[dy + r];
                    for (int dx = -r; dx <= r; ++dx) {
                        int xx = clampi(x + dx, 0, w - 1);
                        acc += (wy * wt[dx + r]) * (float)in[((size_t)yy * w + xx) * 3 + c];
                    }
                }
                out[((size_t)y * w + x) * 3 + c] = to_u8(acc);
            }
        }
    }
}

void cpu_sobel(const uint8_t* in, uint8_t* out, int w, int h, int threads) {
    const int nt = std::max(1, threads);
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int l[9];
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const uint8_t* p =
                        in + ((size_t)clampi(y + dy, 0, h - 1) * w + clampi(x + dx, 0, w - 1)) * 3;
                    l[(dy + 1) * 3 + (dx + 1)] = luma(p[0], p[1], p[2]);
                }
            uint8_t v = sobel_from_luma(l);
            uint8_t* o = out + ((size_t)y * w + x) * 3;
            o[0] = o[1] = o[2] = v;
        }
    }
}

void cpu_median(const uint8_t* in, uint8_t* out, int w, int h, int threads) {
    const int nt = std::max(1, threads);
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                uint8_t v[9];
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        v[(dy + 1) * 3 + (dx + 1)] =
                            in[((size_t)clampi(y + dy, 0, h - 1) * w + clampi(x + dx, 0, w - 1)) * 3 + c];
                out[((size_t)y * w + x) * 3 + c] = median9(v);
            }
        }
    }
}
