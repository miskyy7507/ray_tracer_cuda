#include <iostream>
#include <vector>
#include <cmath>

#include "Vector3.cuh"
#include "Ray.cuh"
#include "export_framebuffer_to_bitmap.h"


#define checkCudaErrors(val) cuda_err_chk( (val), #val, __FILE__, __LINE__ )
void cuda_err_chk(cudaError_t code, const char* const func, const char* const file, int const line) {
    if (code != cudaSuccess) {
        std::cerr << "Błąd CUDA nr " << (code) << " (" << cudaGetErrorString(code) << ")\n" \
        << " w pliku " << file << ":" << line << " funkcja '" << func << "'\n";
        cudaDeviceReset();
        exit(127);
    }
}

__device__ float hit_sphere(const Vector3& center, const float radius, const Ray& ray) {
    Vector3 oc = center - ray.origin(); // co to jest?

    float a = ray.direction().dot(ray.direction());
    // float b = -2.0f * ray.direction().dot(oc);
    float h = ray.direction().dot(oc);
    float c = oc.dot(oc) - radius * radius;

    // float discriminant = b*b - 4*a*c;
    float discriminant = h*h - a*c;

    if (discriminant < 0) {
        return -1.0f;
    } else {
        // return (-b - sqrtf(discriminant) ) / (2.0f*a);
        return (h - sqrtf(discriminant)) / a;
    }
}

__device__ Vector3 get_ray_color(const Ray& r) {
    // if (hit_sphere(Vector3(0.0f, 0.0f, -1.0f), 0.5f, r)) {
    //     return Vector3(0.4f, 0.0f, 0.0f);
    // }
    Vector3 sphere_center = Vector3(0.0f, 0.0f, -1.0f);
    float sphere_radius = 0.5f;
    float t = hit_sphere(sphere_center, sphere_radius, r);
    if (t > 0.0f) {
        // Vector3 normal = (r.point_at(t) - sphere_center).normalized();
        Vector3 normal = (r.point_at(t) - sphere_center) / sphere_radius;
        return (normal + Vector3(1.0f, 1.0f, 1.0f)) * 0.5f;
    }

    auto unit_dir = r.direction().normalized();
    float a = 0.5f * (unit_dir.y + 1.0f);
    return (Vector3(1.0f, 1.0f, 1.0f) * (1.0f-a) + Vector3(0.5f, 0.7f, 1.0f) * a);
}

__global__ void render(Vector3 *fb, unsigned int width, unsigned int height,
    Vector3 pixel_delta_x, Vector3 pixel_delta_y, Vector3 viewport_upper_left_pixel_center, Vector3 cam_center) {

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned int pixel_index = y * width + x;


    auto pixel_center = viewport_upper_left_pixel_center + (pixel_delta_x * x) + (pixel_delta_y * y);
    auto ray_dir = pixel_center - cam_center;

    auto r = Ray(cam_center, ray_dir);
    auto color = get_ray_color(r);

    fb[pixel_index] = color;
}


int main() {
    // int image_height = 2880;
    int image_height = 720;
    float aspect_ratio = 16.0 / 9.0;
    int image_width =  aspect_ratio * image_height;

    int fb_size = image_height * image_width;

    float focal_length = 1.0f;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * (float)image_width / image_height;
    auto camera_center = Vector3();

    // wektory przebiegające wzdłuż szerokości i wysokości viewportu
    auto viewport_x = Vector3(viewport_width, 0.0f, 0.0f);
    auto viewport_y = Vector3(0.0f, -viewport_height, 0.0f);

    // wektory o długości przestrzeni między środkami pikseli kamery
    auto pixel_delta_x = viewport_x / image_width;
    auto pixel_delta_y = viewport_y / image_height;

    // oblicz położenie środka lewego górnego piksela
    auto viewport_upper_left_corner = camera_center
        - Vector3(0.0f, 0.0f, focal_length)
        - viewport_x/2.0f - viewport_y/2.0f;
    auto viewport_upper_left_pixel_center = viewport_upper_left_corner + (pixel_delta_x + pixel_delta_y) * 0.5f;

    auto fb_host = std::vector<Vector3>(fb_size);

    Vector3 *fb_device;
    checkCudaErrors(cudaMalloc(&fb_device, sizeof(Vector3) * fb_size));

    int block_x_size = 8;
    int block_y_size = 8;

    dim3 threads_per_block(block_x_size,block_y_size);
    dim3 blocks_per_grid(image_width / block_x_size+1,image_height / block_y_size+1); // TODO: how does this even work

    render<<<blocks_per_grid, threads_per_block>>>(fb_device, image_width, image_height, pixel_delta_x, pixel_delta_y, viewport_upper_left_pixel_center ,camera_center);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(fb_host.data(), fb_device, sizeof(Vector3) * fb_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(fb_device));
    checkCudaErrors(cudaDeviceSynchronize());

    export_framebuffer_to_bitmap(fb_host, image_width, image_height, "image.bmp");

    return 0;
}
