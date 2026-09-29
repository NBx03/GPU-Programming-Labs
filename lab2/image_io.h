#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;  // w*h*3, RGB 8 бит
};

// Читает PNG любого типа (gray/palette/RGB/RGBA, 1..16 бит) и приводит к RGB 8 бит (альфа отбрасывается).
bool read_png(const std::string& path, Image& img, std::string& err);
// Пишет RGB 8 бит.
bool write_png(const std::string& path, const Image& img, std::string& err);
// Детерминированное тестовое изображение (градиенты, окружности, полосы, шум).
Image make_test_image(int w, int h);
// Масштабирование "ближайший сосед" -- для бенчмарка разных размеров.
Image resize_nearest(const Image& src, int w, int h);
