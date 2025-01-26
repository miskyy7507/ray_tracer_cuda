#ifndef METAL_CUH
#define METAL_CUH
#include "Material.cuh"

class Metal: public Material {
public:
    __device__ Metal(const Vector3& _color, float _roughness);

    __device__ bool scatter(const Ray &r_in, const HitRecord &rec, Vector3 &attenuation, Ray &scattered,
        curandState *local_random_state) const override;

private:
    Vector3 color;
    float roughness;
};

__global__ void create_metal(Vector3 color, float roughness, Material **list, size_t index);

#endif //METAL_CUH
