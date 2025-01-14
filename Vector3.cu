#include "Vector3.cuh"
#include <cmath>

__device__ __host__ Vector3::Vector3() : x{0.0f}, y{0.0f}, z{0.0f} {}

__device__ __host__ Vector3::Vector3(float _x, float _y, float _z) : x{_x}, y{_y}, z{_z}  {}

__device__ __host__ Vector3 Vector3::operator-() const {
    return {-x, -y, -z};
}

__device__ __host__ Vector3 Vector3::operator+(const Vector3 &v) const {
    return {x + v.x, y + v.y, z + v.z};
}

__device__ __host__ Vector3 Vector3::operator-(const Vector3 &v) const {
    return {x - v.x, y - v.y, z - v.z};
}

__device__ __host__  Vector3 Vector3::operator*(const Vector3 &v) const {
    return {x * v.x, y * v.y, z * v.z};
}

__device__ __host__ Vector3 Vector3::operator*(const float &n) const {
    return {x*n, y*n, z*n};
}

__device__ __host__ Vector3 Vector3::operator/(const float &n) const {
    return (*this * (1.0f / n));
}

__device__ __host__ float Vector3::dot(const Vector3 &v) const {
    return x*v.x + y*v.y + z*v.z;
}

__device__ __host__ Vector3 Vector3::cross(const Vector3 &v) const {
    return {
        y*v.z - z*v.y,
        z*v.x - x*v.z,
        x*v.y - y*v.x
    };
}

__device__ __host__ float Vector3::length_squared() const {
    return x*x + y*y + z*z;
}

__device__ __host__ float Vector3::length() const {
    return std::sqrt(length_squared());
}

__device__ __host__ Vector3 Vector3::normalized() const {
    // return *this / length();
    return this->operator/(this->length());
}
