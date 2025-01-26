#ifndef MATTE_CUH
#define MATTE_CUH
#include "Material.cuh"

class Matte: public Material {
public:
    __device__ Matte(const Vector3& _color);

    __device__ bool scatter(const Ray &r_in, const HitRecord &rec, Vector3 &attenuation, Ray &scattered, curandState* local_random_state) const override;

private:
    Vector3 color;
};

__global__ void create_matte(Vector3 color, Material **list, size_t index);

#endif //MATTE_CUH
