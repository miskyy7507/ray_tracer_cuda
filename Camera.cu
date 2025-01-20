#include "Camera.cuh"

#include "Material.cuh"
#include "rtcuda.h"

__host__ __device__ Camera::Camera(
    const int _image_height,
    const float _aspect_ratio,
    Hittable** _world,
    curandState* _curand_state
) : image_height(_image_height), aspect_ratio(_aspect_ratio), world(_world), curand_state(_curand_state)
{
    this->image_width = aspect_ratio * image_height;

    this->sample_count = 256;

    float focal_length = 1.0f;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * image_width / image_height;
    this->camera_center = Vector3(); // 0,0,0

    // wektory przebiegające wzdłuż szerokości i wysokości viewportu
    auto viewport_x = Vector3(viewport_width, 0.0f, 0.0f);
    auto viewport_y = Vector3(0.0f, -viewport_height, 0.0f);

    this->pixel_delta_x = viewport_x / image_width;
    this->pixel_delta_y = viewport_y / image_height;

    auto viewport_upper_left_corner = camera_center
        - Vector3(0.0f, 0.0f, focal_length)
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

    for (int i = 0; i < depth; i++) {
        HitRecord rec;
        if ((*this->world)->hit(current_ray, 0.0001f, RTCuda::INF, rec)) {
            // Vector3 dir = Vector3::random_unit_vector(local_random_state) + rec.normal;
            // current_attenuation *= 0.5f;
            // current_ray = Ray(rec.point, dir);
            Ray scattered;
            Vector3 attenuate;
            if (rec.material->scatter(current_ray, rec, attenuate, scattered, local_random_state)) {
                current_attenuation = current_attenuation * attenuate;
                current_ray = scattered;
            } else {
                return Vector3(0.0f, 0.0f, 0.0f);
            }


        } else {
            auto unit_dir = current_ray.direction().normalized();
            float a = 0.5f * (unit_dir.y + 1.0f);
            Vector3 c = (Vector3(1.0f, 1.0f, 1.0f) * (1.0f-a) + Vector3(0.5f, 0.7f, 1.0f) * a);
            return c * current_attenuation;
        }
    }

    return Vector3(0.0f, 0.0f, 0.0f); // przekroczono głębokość rekursji
}
