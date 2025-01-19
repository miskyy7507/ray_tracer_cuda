#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <curand_kernel.h>

#include "Camera.cuh"
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

__global__ void random_state_init(int width, int height, curandState *rand_state) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;
    unsigned int pixel_index = y * width + x;

    curand_init(1+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void create_world(
    const size_t list_size,
    Hittable** list,
    Hittable** world
    ) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    list[0] = new Sphere(Vector3(0.0f, 0.0f, -1.0f), 0.5f);
    list[1] = new Sphere(Vector3(0.0f, -100.5f, 1.0f), 100.f);
    *world = new HittableList(list, 2);
}

__global__ void destroy_world(const size_t list_size, Hittable** list, Hittable** world) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (size_t i = 0; i < list_size; i++) {
        delete list[i];
    }
    delete *world;
}

__global__ void render(unsigned int width, unsigned int height, Camera* camera, Vector3* buffer) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    buffer[x + y * width] = camera->render_pixel(x, y);
}


int main() {
    int image_height = 720;
    float aspect_ratio = 16.0 / 9.0;
    int image_width =  aspect_ratio * image_height;

    int buffer_size = image_height * image_width;

    auto fb_host = std::vector<Vector3>(buffer_size);

    int block_x_size = 8;
    int block_y_size = 8;
    dim3 threads_per_block(block_x_size,block_y_size);
    dim3 blocks_per_grid(
        static_cast<int>(std::ceil(static_cast<float>(image_width) / block_x_size)),
        static_cast<int>(std::ceil(static_cast<float>(image_height) / block_y_size))
    );

    // alokacja pamięci gpu
    Vector3* d_buffer;
    checkCudaErrors(cudaMalloc(&d_buffer, sizeof(Vector3) * buffer_size));
    checkCudaErrors(cudaDeviceSynchronize());

    constexpr size_t list_size = 2;
    Hittable** d_hitlist;
    checkCudaErrors(cudaMalloc(&d_hitlist, sizeof(Hittable*) * list_size));
    checkCudaErrors(cudaDeviceSynchronize());
    Hittable** d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable*)));
    checkCudaErrors(cudaDeviceSynchronize());
    create_world<<<1,1>>>(list_size, d_hitlist, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // stan cuRAND (generator liczb pseudolosowych)
    curandState *d_random_state;
    checkCudaErrors(cudaMalloc(&d_random_state, buffer_size * sizeof(curandState)));
    checkCudaErrors(cudaDeviceSynchronize());
    random_state_init<<<blocks_per_grid, threads_per_block>>>(image_width, image_height, d_random_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Camera* d_cam;
    checkCudaErrors(cudaMalloc(&d_cam, sizeof(Camera)));
    auto camera = Camera(image_height, aspect_ratio, d_world, d_random_state);
    checkCudaErrors(cudaMemcpy(d_cam, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

    render<<<blocks_per_grid, threads_per_block>>>(image_width, image_height, d_cam, d_buffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // kopiowanie gotowego bufora obrazu z gpu do hosta
    checkCudaErrors(cudaMemcpy(fb_host.data(), d_buffer, sizeof(Vector3) * buffer_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    // zwolnienie pamięci GPU
    destroy_world<<<1,1>>>(list_size, d_hitlist, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(d_buffer));
    checkCudaErrors(cudaFree(d_hitlist));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_random_state));
    checkCudaErrors(cudaDeviceSynchronize());

    export_framebuffer_to_bitmap(fb_host, image_width, image_height, "image.bmp");

    return 0;
}
