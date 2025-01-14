//
// Created by miskyy on 11.01.25.
//

#include "Ray.cuh"

__device__ Ray::Ray() : orig(Vector3()), dir(Vector3()) {} // todo: see if without initializer list is faster

__device__ Ray::Ray(const Vector3 &_orig, const Vector3 &_dir) : orig(_orig), dir(_dir) {}

__device__ const Vector3 & Ray::origin() const {
    return orig;
}

__device__ const Vector3 & Ray::direction() const {
    return dir;
}

__device__ Vector3 Ray::point_at(float t) const {
    return orig + dir * t;
}
