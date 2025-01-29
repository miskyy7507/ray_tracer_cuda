#ifndef LIGHT_CUH
#define LIGHT_CUH
#include "Material.cuh"

class Light: public Material {
public:
    __device__ Light(const Vector3& _color);

    __device__ Vector3 emitted_color() const override;

private:
    Vector3 color;
};

__global__ void create_light(Vector3 color, Material **list, size_t index);

#endif //LIGHT_CUH
