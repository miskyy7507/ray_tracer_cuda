#ifndef METAL_CUH
#define METAL_CUH
#include "Material.cuh"

class Metal: public Material {
public:
    __device__ Metal(const Vector3& _color, float _fuzz);

    __device__ bool scatter(const Ray &r_in, const HitRecord &rec, Vector3 &attenuation, Ray &scattered,
        curandState *local_random_state) const override;

private:
    Vector3 color;
    float fuzziness;
};

#endif //METAL_CUH
