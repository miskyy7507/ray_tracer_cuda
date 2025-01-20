#include "Metal.cuh"

__device__ Metal::Metal(const Vector3 &_color, float _fuzz) : color(_color), fuzziness(_fuzz < 1 ? _fuzz : 1) {}

__device__ bool Metal::scatter(const Ray &r_in, const HitRecord &rec, Vector3 &attenuation, Ray &scattered, curandState *local_random_state) const {
    auto dir = r_in.direction().reflect(rec.normal);
    if (fuzziness > 0.0f) {
        dir = dir.normalized() + Vector3::random_unit_vector(local_random_state) * fuzziness;
    }
    // auto dir = r_in.direction().reflect(rec.normal).normalized() + (Vector3::random_unit_vector(local_random_state) * fuzziness);
    scattered = Ray(rec.point, dir);
    attenuation = this->color;
    return scattered.direction().dot(rec.normal) > 0;
}
