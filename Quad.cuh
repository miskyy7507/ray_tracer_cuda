#ifndef QUAD_CUH
#define QUAD_CUH
#include "Hittable.cuh"
#include "Vector3.cuh"

class Material;

class Quad: public Hittable {
public:
    __device__ Quad(const Vector3& Q, const Vector3& _u, const Vector3& _v, Material* _material);

    __device__ bool hit(const Ray &r, Interval ray_t, HitRecord &rec) const override;

private:
    Vector3 Q, u, v, w;
    Material* material;
    Vector3 normal;
    float D;
};

__global__ void create_quad(Vector3 Q, Vector3 u, Vector3 v, int mat_index, Material** mat_list, Hittable** list, size_t index);

#endif //QUAD_CUH
