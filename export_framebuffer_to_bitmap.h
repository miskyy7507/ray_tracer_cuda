#ifndef EXPORT_FRAMEBUFFER_TO_BITMAP_H
#define EXPORT_FRAMEBUFFER_TO_BITMAP_H

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>

#include "stb_image_write.h"

#include <vector>

#include "Vector3.cuh"

void export_framebuffer_to_bitmap(const std::vector<Vector3>& fb, const int width, const int height, const char* filename) {
    unsigned int buffer_size = width * height * 3;
    std::vector<unsigned char> imageData(buffer_size);

    for (int i = 0; i < width*height; i++) {
        imageData[i * 3 + 0] = 255.999 * fb[i].x; // Red
        imageData[i * 3 + 1] = 255.999 * fb[i].y; // Green
        imageData[i * 3 + 2] = 255.999 * fb[i].z; // Blue
    }

    if (stbi_write_bmp(filename, width, height, 3, imageData.data())) {
        std::clog << "Plik graficzny pomyślnie zapisano do: " << filename << std::endl;
    } else {
        std::clog << "Błąd zapisu pliku: " << filename << std::endl;
    }
}

#endif //EXPORT_FRAMEBUFFER_TO_BITMAP_H
