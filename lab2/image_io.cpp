#include "image_io.h"

#include <png.h>

#include <cmath>
#include <cstdio>
#include <cstring>

bool read_png(const std::string& path, Image& img, std::string& err) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        err = "cannot open " + path;
        return false;
    }
    unsigned char sig[8];
    if (std::fread(sig, 1, 8, fp) != 8 || png_sig_cmp(sig, 0, 8) != 0) {
        std::fclose(fp);
        err = path + " is not a PNG file";
        return false;
    }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png ? png_create_info_struct(png) : nullptr;
    if (!png || !info) {
        std::fclose(fp);
        err = "libpng init failed";
        return false;
    }
    std::vector<png_bytep> rows;  // объявлен до setjmp
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        err = "libpng error while reading " + path;
        return false;
    }
    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    png_uint_32 w = png_get_image_width(png, info), h = png_get_image_height(png, info);
    int color = png_get_color_type(png, info), depth = png_get_bit_depth(png, info);
    if (depth == 16) png_set_strip_16(png);
    if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    png_set_strip_alpha(png);
    png_set_interlace_handling(png);
    png_read_update_info(png, info);

    if (png_get_rowbytes(png, info) != (size_t)w * 3) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        err = "unexpected pixel format in " + path;
        return false;
    }
    img.w = (int)w;
    img.h = (int)h;
    img.rgb.assign((size_t)w * h * 3, 0);
    rows.resize(h);
    for (png_uint_32 y = 0; y < h; ++y) rows[y] = img.rgb.data() + (size_t)y * w * 3;
    png_read_image(png, rows.data());
    png_read_end(png, nullptr);
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);
    return true;
}

bool write_png(const std::string& path, const Image& img, std::string& err) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        err = "cannot open " + path + " for writing";
        return false;
    }
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png ? png_create_info_struct(png) : nullptr;
    if (!png || !info) {
        std::fclose(fp);
        err = "libpng init failed";
        return false;
    }
    std::vector<png_bytep> rows;
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        std::fclose(fp);
        err = "libpng error while writing " + path;
        return false;
    }
    png_init_io(png, fp);
    png_set_IHDR(png, info, img.w, img.h, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    rows.resize(img.h);
    for (int y = 0; y < img.h; ++y)
        rows[y] = const_cast<png_bytep>(img.rgb.data() + (size_t)y * img.w * 3);
    png_write_image(png, rows.data());
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    std::fclose(fp);
    return true;
}

Image make_test_image(int w, int h) {
    Image im;
    im.w = w;
    im.h = h;
    im.rgb.resize((size_t)w * h * 3);
    uint32_t seed = 12345;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (int)((seed >> 16) & 0xFF);
    };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double u = (double)x / w, v = (double)y / h;
            double r = 255 * u, g = 255 * v, b = 255 * (1.0 - 0.5 * (u + v));
            // концентрические окружности и диагональные полосы -- чёткие границы для Собеля
            double d = std::hypot(u - 0.5, v - 0.5);
            if (std::fmod(d * 12, 1.0) < 0.5) {
                r = 255 - r;
                b = 255 - b;
            }
            if (std::fmod((u + v) * 20, 1.0) < 0.15) g = 255 - g;
            // слабый шум + "соль и перец" ~3% (для медианного фильтра)
            int n = rnd();
            uint8_t* p = &im.rgb[((size_t)y * w + x) * 3];
            p[0] = (uint8_t)std::fmin(255.0, std::fmax(0.0, r + (n - 128) / 16.0));
            p[1] = (uint8_t)std::fmin(255.0, std::fmax(0.0, g + (n - 128) / 16.0));
            p[2] = (uint8_t)std::fmin(255.0, std::fmax(0.0, b + (n - 128) / 16.0));
            if (rnd() < 8) p[0] = p[1] = p[2] = (rnd() & 1) ? 255 : 0;
        }
    return im;
}

Image resize_nearest(const Image& src, int w, int h) {
    Image dst;
    dst.w = w;
    dst.h = h;
    dst.rgb.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y) {
        int sy = (int)((long long)y * src.h / h);
        for (int x = 0; x < w; ++x) {
            int sx = (int)((long long)x * src.w / w);
            std::memcpy(&dst.rgb[((size_t)y * w + x) * 3], &src.rgb[((size_t)sy * src.w + sx) * 3], 3);
        }
    }
    return dst;
}
