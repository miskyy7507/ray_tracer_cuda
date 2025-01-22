#ifndef VECTOR3_CUH
#define VECTOR3_CUH
#include <curand_kernel.h>

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

    // Wektor o losowych punktach x, y, z w przedziale (0.0, 1.0).
    __device__ static Vector3 random(curandState* local_random_state);

    // Wektor o losowych punktach x, y, z w przedziale (min, max).
    __device__ static Vector3 random(float min, float max, curandState* local_random_state);

    // Funkcja zwracająca nowy wektor o znormalizowanej długości w losowym kierunku.
    // Tzn. wektor na powierzchni sfery o promieniu 1.
    __device__ static Vector3 random_unit_vector_old(curandState* local_random_state);

    __device__ static Vector3 random_unit_vector(curandState* local_random_state);

    __device__ Vector3 reflect(const Vector3& normal) const;
};


#endif //VECTOR3_CUH
