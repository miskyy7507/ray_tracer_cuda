#ifndef MATERIAL_CUH
#define MATERIAL_CUH
#include "Hittable.cuh"
#include "Ray.cuh"

class Material {
public:
    __device__ virtual ~Material() {}

    __device__ virtual bool scatter(
        const Ray& r_in,
        const HitRecord& rec,
        Vector3& attenuation,
        Ray& scattered,
        curandState* local_random_state
    ) const {
        return false;
    }
};

#endif //MATERIAL_CUH
