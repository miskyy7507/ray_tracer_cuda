#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "Interval.cuh"
#include "Ray.cuh"
#include "Vector3.cuh"

class Material;

class HitRecord {
public:
    Vector3 point;
    Vector3 normal;
    Material* material;
    float t;
};

class Hittable {
public:
    __device__ virtual ~Hittable() {}

    __device__ virtual bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const = 0;
};

#endif //HITTABLE_CUH
