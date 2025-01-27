#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "Hittable.cuh"
#include "Vector3.cuh"

class Sphere: public Hittable {
public:
    __device__ Sphere(const Vector3& _center, float _radius, Material* _material);

    __device__ bool hit(const Ray &r, Interval ray_t, HitRecord &rec) const override;

private:
    Vector3 center;
    float radius;
    Material* material;
};

__global__ void create_sphere(Vector3 center, float radius, int mat_index, Material** mat, Hittable** list, size_t index);

#endif //SPHERE_CUH
