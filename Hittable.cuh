#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "Ray.cuh"
#include "Vector3.cuh"

struct HitRecord {
    Vector3 point;
    Vector3 normal;
    float t;
};

class Hittable {
public:
    __device__ virtual ~Hittable() {}

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};

#endif //HITTABLE_CUH
