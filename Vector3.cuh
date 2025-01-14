#ifndef VECTOR3_CUH
#define VECTOR3_CUH

class Vector3 {
public:
    float x, y, z;

    // "pusty" wektor (0.0, 0.0, 0.0)
    __device__ __host__ Vector3();
    __device__ __host__ Vector3(float _x, float _y, float _z);

    // wektor przeciwny
    __device__ __host__ Vector3 operator-() const;

    // __device__ __host__ Vector3 operator+=(const Vector3& v) const;
    // __device__ __host__ Vector3 operator*=(const Vector3& v) const;

    // suma wektorów
    __device__ __host__ Vector3 operator+(const Vector3& v) const;

    // różnica wektorów
    __device__ __host__ Vector3 operator-(const Vector3& v) const;

    // iloczyn elementarny (Hadamarda) wektorów
    __device__ __host__ Vector3 operator*(const Vector3& v) const;

    // mnożenie wektora przez skalar
    __device__ __host__ Vector3 operator*(const float& n) const;

    // dzielenie wektora przez skalar
    __device__ __host__ Vector3 operator/(const float& n) const;

    // iloczyn skalarny
    __device__ __host__ float dot(const Vector3& v) const;

    // iloczyn wektorowy
    __device__ __host__ Vector3 cross(const Vector3& v) const;

    // długość wektora do kwadratu
    __device__ __host__ float length_squared() const;

    // długość wektora
    __device__ __host__ float length() const;

    // wektor znormalizowany (o długości 1)
    __device__ __host__ Vector3 normalized() const;
};

#endif //VECTOR3_CUH
