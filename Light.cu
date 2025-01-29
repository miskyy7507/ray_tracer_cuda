#include "Light.cuh"

__device__ Light::Light(const Vector3 &_color) : color(_color) {}

__device__ Vector3 Light::emitted_color() const {
    return this->color;
}

__global__ void create_light(Vector3 color, Material **list, size_t index) {
    list[index] = new Light(color);
}