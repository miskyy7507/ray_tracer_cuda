#ifndef CAMERA_CUH
#define CAMERA_CUH
#include <curand_kernel.h>

#include "Hittable.cuh"
#include "Vector3.cuh"

class Camera {
public:
    __host__ __device__  Camera(
        int _image_height,
        float _aspect_ratio,
        float _vfov_deg,
        Vector3 look_from,
        Vector3 look_at,
        Hittable** _world,
        curandState* _curand_state
    );

    __device__ Vector3 render_pixel(int x, int y);

private:
    __device__ Vector3 get_ray_color(const Ray& r, int depth, curandState* local_random_state);

    int image_width, image_height;
    int sample_count;                           // ilość próbek na jeden piksel
    float aspect_ratio;
    Vector3 camera_center;
    Vector3 pixel_delta_x, pixel_delta_y;       // wektory o długości przestrzeni między środkami pikseli kamery
    Vector3 viewport_upper_left_pixel_center;   // położenie środka lewego górnego piksela
    Hittable** world;
    curandState* curand_state;
};

#endif //CAMERA_CUH
