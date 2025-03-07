/**
 * @file Camera.cu
 * @brief Implementacja klasy Camera.
 */
#include "Camera.cuh"
#include "Material.cuh"
#include "rtcuda.h"

__host__ __device__ Camera::Camera(
    const int _image_height,
    const float _aspect_ratio,
    float _vfov_deg,
    int _sample_count,
    Vector3 look_from,
    Vector3 look_at,
    Vector3 _background_color,
    Hittable** _world,
    curandState* _curand_state
) : image_height(_image_height), sample_count(_sample_count), aspect_ratio(_aspect_ratio), background_color(_background_color), world(_world), curand_state(_curand_state)
{
    this->image_width = aspect_ratio * image_height;

    float vfov_rad = _vfov_deg * (RTCuda::PI / 180);

    float focal_length = (look_from - look_at).length();
    float h = tanf(vfov_rad/2);
    float viewport_height = 2.0f * h * focal_length;
    float viewport_width = viewport_height * image_width / image_height;
    this->camera_center = look_from;

    Vector3 vup = Vector3(0.0f, 1.0f, 0.0f);

    Vector3 w = (look_from - look_at) / focal_length;
    Vector3 u = vup.cross(w).normalized();
    Vector3 v = w.cross(u).normalized();

    // wektory przebiegające wzdłuż szerokości i wysokości viewportu
    auto viewport_x = u * viewport_width;
    auto viewport_y = -v * viewport_height;

    this->pixel_delta_x = viewport_x / image_width;
    this->pixel_delta_y = viewport_y / image_height;

    auto viewport_upper_left_corner = camera_center
        - (w * focal_length)
        - viewport_x/2.0f - viewport_y/2.0f;
    this->viewport_upper_left_pixel_center =
        viewport_upper_left_corner + (pixel_delta_x + pixel_delta_y) * 0.5f;
}

__device__ Vector3 Camera::render_pixel(int x, int y) {
    unsigned int pixel_index = y * this->image_width + x;
    curandState* local_random_state = &this->curand_state[pixel_index];

    Vector3 pixel_color = Vector3();
    for (int i = 0; i < this->sample_count; i++) {
        float random_delta_x = curand_uniform(local_random_state) - 0.5f;
        float random_delta_y = curand_uniform(local_random_state) - 0.5f;

        Vector3 pixel_sample = viewport_upper_left_pixel_center
            + pixel_delta_x * (x + random_delta_x)
            + pixel_delta_y * (y + random_delta_y);

        auto ray_dir = pixel_sample - this->camera_center;

        auto r = Ray(this->camera_center, ray_dir);
        pixel_color = pixel_color + get_ray_color(r, 50, local_random_state);
    }

    return pixel_color / sample_count;
}

__device__ Vector3 Camera::get_ray_color(const Ray& r, int depth, curandState* local_random_state) {
    Ray current_ray = r;
    Vector3 current_attenuation = Vector3(1.0f, 1.0f, 1.0f);
    Vector3 accumulated_color = Vector3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < depth; i++) {
        HitRecord rec;
        if ((*this->world)->hit(current_ray, Interval(0.0005f, RTCuda::INF), rec)) {
            accumulated_color = accumulated_color + rec.material->emitted_color() * current_attenuation;
            Ray scattered;
            Vector3 attenuate;
            if (rec.material->scatter(current_ray, rec, attenuate, scattered, local_random_state)) {
                current_attenuation = current_attenuation * attenuate;
                current_ray = scattered;
            } else {
                return accumulated_color;
            }


        } else {
            return this->background_color * current_attenuation;
        }
    }

    return Vector3(0.0f, 0.0f, 0.0f); // przekroczono głębokość rekursji
}
