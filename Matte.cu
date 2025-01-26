#include "Matte.cuh"

__device__ Matte::Matte(const Vector3 &_color) : color(_color) {}

__device__ bool Matte::scatter(const Ray& r_in, const HitRecord& rec, Vector3& attenuation, Ray& scattered, curandState* local_random_state) const {
    auto dir = Vector3::random_unit_vector(local_random_state) + rec.normal;
    scattered = Ray(rec.point, dir);
    attenuation = this->color;
    return true;
}

__global__ void create_matte(Vector3 color, Material **list, size_t index) {
    list[index] = new Matte(color);
}