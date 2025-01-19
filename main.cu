#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <curand_kernel.h>

#include "Vector3.cuh"
#include "Ray.cuh"
#include "export_framebuffer_to_bitmap.h"
#include "Hittable.cuh"
#include "HittableList.cuh"
#include "Sphere.cuh"


#define checkCudaErrors(val) cuda_err_chk( (val), #val, __FILE__, __LINE__ )
void cuda_err_chk(cudaError_t code, const char* const func, const char* const file, int const line) {
    if (code != cudaSuccess) {
        std::cerr << "Błąd CUDA nr " << (code) << " (" << cudaGetErrorString(code) << ")\n" \
        << " w pliku " << file << ":" << line << " funkcja '" << func << "'\n";
        cudaDeviceReset();
        exit(127);
    }
}

__device__ Vector3 get_ray_color(const Ray& r, Hittable** world) {
    // Vector3 sphere_center = Vector3(0.0f, 0.0f, -1.0f);
    // float sphere_radius = 0.5f;
    // float t = hit_sphere(sphere_center, sphere_radius, r);
    // if (t > 0.0f) {
    //     // Vector3 normal = (r.point_at(t) - sphere_center).normalized();
    //     Vector3 normal = (r.point_at(t) - sphere_center) / sphere_radius;
    //     return (normal + Vector3(1.0f, 1.0f, 1.0f)) * 0.5f;
    // }
    HitRecord rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return (rec.normal + Vector3(1.0f, 1.0f, 1.0f)) * 0.5f;
    }

    auto unit_dir = r.direction().normalized();
    float a = 0.5f * (unit_dir.y + 1.0f);
    return (Vector3(1.0f, 1.0f, 1.0f) * (1.0f-a) + Vector3(0.5f, 0.7f, 1.0f) * a);
}

__global__ void render_init(int width, int height, curandState *rand_state) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;
    unsigned int pixel_index = y * width + x;

    curand_init(2137, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(Vector3 *fb, unsigned int width, unsigned int height,
    Vector3 pixel_delta_x, Vector3 pixel_delta_y, Vector3 viewport_upper_left_pixel_center, Vector3 cam_center, Hittable** world, const curandState *rand_state, int num_samples = 512) {

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned int pixel_index = y * width + x;
    curandState local_random_state = rand_state[pixel_index];

    // Vector3 pixel_sample = viewport_upper_left_pixel_center + (pixel_delta_x * x) + (pixel_delta_y * y);
    // auto ray_dir = pixel_center - cam_center;

    // auto r = Ray(cam_center, ray_dir);
    // auto color = get_ray_color(r, world);

    Vector3 pixel_color = Vector3();
    for (int i = 0; i < num_samples; i++) {
        float random_delta_x = curand_uniform(&local_random_state) - 0.5f;
        float random_delta_y = curand_uniform(&local_random_state) - 0.5f;

        Vector3 pixel_sample = viewport_upper_left_pixel_center
            + pixel_delta_x * (x + random_delta_x)
            + pixel_delta_y * (y + random_delta_y);

        auto ray_dir = pixel_sample - cam_center;

        auto r = Ray(cam_center, ray_dir);
        pixel_color = pixel_color + get_ray_color(r, world);
    }

    fb[pixel_index] = pixel_color / num_samples;
}

__global__ void create_world(Hittable** list, Hittable** world) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    list[0] = new Sphere(Vector3(0.0f, 0.0f, -1.0f), 0.5f);
    list[1] = new Sphere(Vector3(0.0f, -100.5f, 1.0f), 100.f);
    *world = new HittableList(list, 2);
}

__global__ void delete_world(Hittable** list, const size_t l, Hittable** world) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (size_t i = 0; i < l; i++) {
        delete list[i];
    }
    delete *world;
}


int main() {
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

    // alokacja pamięci gpu
    Vector3 *fb_device;
    checkCudaErrors(cudaMalloc(&fb_device, sizeof(Vector3) * fb_size));

    constexpr size_t list_size = 2;
    Hittable **list_device;
    checkCudaErrors(cudaMalloc((void**)&list_device, sizeof(Hittable*) * list_size));

    Hittable **world_device;
    checkCudaErrors(cudaMalloc((void**)&world_device, sizeof(Hittable*)));

    create_world<<<1,1>>>(list_device, world_device);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // stan cuRAND (generator liczb pseudolosowych)
    curandState *rand_state_device;
    checkCudaErrors(cudaMalloc(&rand_state_device, fb_size * sizeof(curandState)));

    int block_x_size = 8;
    int block_y_size = 8;

    dim3 threads_per_block(block_x_size,block_y_size);
    dim3 blocks_per_grid(
        static_cast<int>(std::ceil(static_cast<float>(image_width) / block_x_size)),
        static_cast<int>(std::ceil(static_cast<float>(image_height) / block_y_size))
    );

    render_init<<<blocks_per_grid, threads_per_block>>>(image_width, image_height, rand_state_device);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks_per_grid, threads_per_block>>>(fb_device, image_width, image_height, pixel_delta_x, pixel_delta_y, viewport_upper_left_pixel_center ,camera_center, world_device, rand_state_device);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // kopiowanie gotowego bufora obrazu z gpu do hosta
    checkCudaErrors(cudaMemcpy(fb_host.data(), fb_device, sizeof(Vector3) * fb_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    // cleanup
    delete_world<<<1,1>>>(list_device, list_size, world_device);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(fb_device));
    checkCudaErrors(cudaFree(list_device));
    checkCudaErrors(cudaFree(world_device));
    checkCudaErrors(cudaFree(rand_state_device));
    checkCudaErrors(cudaDeviceSynchronize());

    export_framebuffer_to_bitmap(fb_host, image_width, image_height, "image.bmp");

    return 0;
}
