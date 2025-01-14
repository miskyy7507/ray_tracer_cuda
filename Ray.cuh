#ifndef RAY_CUH
#define RAY_CUH
#include "Vector3.cuh"

class Ray {
public:
    __device__ Ray();
    __device__ Ray(const Vector3& _orig, const Vector3& _dir);

    __device__ const Vector3& origin() const;
    __device__ const Vector3& direction() const;

    __device__ Vector3 point_at(float t) const;

private:
    Vector3 orig;
    Vector3 dir;
};

#endif //RAY_CUH
