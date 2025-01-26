#include "Metal.cuh"

__device__ Metal::Metal(const Vector3 &_color, float _roughness) : color(_color), roughness(fminf(_roughness, 1.0f)) {}

__device__ bool Metal::scatter(const Ray &r_in, const HitRecord &rec, Vector3 &attenuation, Ray &scattered, curandState *local_random_state) const {
    // auto dir = r_in.direction().reflect(rec.normal);
    // if (this->roughness > 0.0f) {
    //     dir = dir.normalized() + Vector3::random_unit_vector(local_random_state) * this->roughness;
    // }
    auto dir = r_in.direction().reflect(rec.normal).normalized() + (Vector3::random_unit_vector(local_random_state) * this->roughness);
    scattered = Ray(rec.point, dir);
    attenuation = this->color;
    return scattered.direction().dot(rec.normal) > 0;
}

__global__ void create_metal(Vector3 color, float roughness, Material **list, size_t index) {
    list[index] = new Metal(color, roughness);
}
