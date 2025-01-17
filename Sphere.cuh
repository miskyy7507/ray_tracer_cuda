#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "Hittable.cuh"
#include "Vector3.cuh"

class Sphere: public Hittable {
public:
    __device__ Sphere(const Vector3& _center, float _radius);

    __device__ bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override;

private:
    Vector3 center;
    float radius;
};

#endif //SPHERE_CUH
