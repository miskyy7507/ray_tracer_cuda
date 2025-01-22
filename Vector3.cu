#include "Vector3.cuh"
#include <cmath>
#include <curand_kernel.h>

#include "rtcuda.h"

__device__ __host__ Vector3::Vector3() : x{0.0f}, y{0.0f}, z{0.0f} {}

__device__ __host__ Vector3::Vector3(float _x, float _y, float _z) : x{_x}, y{_y}, z{_z}  {}



__device__ Vector3 Vector3::random(curandState *local_random_state) {
    return {
        curand_uniform(local_random_state),
        curand_uniform(local_random_state),
        curand_uniform(local_random_state)
    };
}

__device__ Vector3 Vector3::random(float min, float max, curandState *local_random_state) {
    return Vector3::random(local_random_state) * (max - min) + Vector3(min, min, min);
}

__device__ Vector3 Vector3::random_unit_vector_old(curandState* local_random_state) {
    Vector3 p;
    float lensq;
    do {
        p = Vector3::random(-1.0f, 1.0f, local_random_state);
    } while ((lensq = p.length_squared()) > 1.0f);
    return p / sqrtf(lensq);
}

// pożyczone z https://github.com/mrdoob/three.js/blob/master/src/math/Vector3.js#L695
// informacja: https://mathworld.wolfram.com/SpherePointPicking.html
__device__ Vector3 Vector3::random_unit_vector(curandState* local_random_state) {
    float theta = curand_uniform(local_random_state) * 2 * RTCuda::PI;
    float u = curand_uniform(local_random_state) * 2.0f - 1.0f;
    float c = sqrtf(1.0f - u * u);

    return {
        c * cosf(theta),
        u,
        c * sinf(theta),
    };
}

__device__ Vector3 Vector3::reflect(const Vector3 &normal) const {
    return *this - normal * this->dot(normal) * 2.0f;
}
