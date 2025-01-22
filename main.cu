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
#include "Matte.cuh"
#include "Metal.cuh"
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
    const size_t hittable_list_size,
    Hittable** hittable_list,
    const size_t material_list_size,
    Material** material_list,
    Hittable** world
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int material_list_index = 0;
    unsigned int hittable_list_index = 0;

    curandState local_random_state;

    curand_init(2137, 0, 0, &local_random_state);

    // ziemia
    material_list[material_list_index] = new Matte(Vector3(0.5f, 0.5f, 0.5f));
    hittable_list[hittable_list_index] = new Sphere(Vector3(0.0f, -1000.0f, -0.0f), 1000.0f, material_list[material_list_index]);

    material_list_index++;
    hittable_list_index++;

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float mat_choice = curand_uniform(&local_random_state);
            Vector3 center = Vector3(0.9f*curand_uniform(&local_random_state)+a, 0.2f, 0.9f*curand_uniform(&local_random_state)+b);

            Vector3 color =
                Vector3(curand_uniform(&local_random_state), curand_uniform(&local_random_state), curand_uniform(&local_random_state)) *
                    Vector3(curand_uniform(&local_random_state), curand_uniform(&local_random_state), curand_uniform(&local_random_state));
            if (mat_choice < 0.8f) {
                material_list[material_list_index] = new Matte(color);
            } else {
                material_list[material_list_index] = new Metal(color, 0.5f*curand_uniform(&local_random_state));
            }
            hittable_list[hittable_list_index] = new Sphere(center, 0.2f, material_list[material_list_index]);

            material_list_index++;
            hittable_list_index++;
        }
    }

    material_list[material_list_index] = new Matte(Vector3(0.1f, 0.2f, 0.5f));                                   // środkowy
    hittable_list[hittable_list_index] = new Sphere(Vector3(0.0f, 1.0f, 0.0f), 1.0f, material_list[material_list_index]);

    material_list_index++;
    hittable_list_index++;
    material_list[material_list_index] = new Metal(Vector3(0.8f, 0.8f, 0.8f), 0.0f);                        // lewy
    hittable_list[hittable_list_index] = new Sphere(Vector3(-4.0f, 1.0f, 0.0f), 1.0f, material_list[material_list_index]);

    material_list_index++;
    hittable_list_index++;
    material_list[material_list_index] = new Metal(Vector3(0.8f, 0.6f, 0.2f), 0.0f);                        // prawy
    hittable_list[hittable_list_index] = new Sphere(Vector3(4.0f, 1.0f, 0.0f), 1.0f, material_list[material_list_index]);

    *world = new HittableList(hittable_list, hittable_list_size);
}

__global__ void destroy_world(
    const size_t hittable_list_size,
    Hittable** hittable_list,
    const size_t material_list_size,
    Material** material_list,
    Hittable** world
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (size_t i = 0; i < hittable_list_size; i++) {
        delete hittable_list[i];
    }
    for (size_t i = 0; i < material_list_size; i++) {
        delete material_list[i];
    }
    delete *world;
}

__global__ void render(unsigned int width, unsigned int height, Camera* camera, Vector3* buffer) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    buffer[x + y * width] = camera->render_pixel(x, y);
}

__global__ void gamma_2_correction(unsigned int width, unsigned int height, Vector3* buffer) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned int pixel_index = x + y * width;

    buffer[pixel_index].x = sqrtf(buffer[pixel_index].x);
    buffer[pixel_index].y = sqrtf(buffer[pixel_index].y);
    buffer[pixel_index].z = sqrtf(buffer[pixel_index].z);
}


int main() {
    int image_height = 720;
    float aspect_ratio = 16.0 / 9.0;
    int image_width =  aspect_ratio * image_height;
    float field_of_view = 20; // pole widzenia wertykalne w kątach

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

    // stan cuRAND (generator liczb pseudolosowych)
    curandState *d_random_state;
    checkCudaErrors(cudaMalloc(&d_random_state, buffer_size * sizeof(curandState)));
    checkCudaErrors(cudaDeviceSynchronize());
    random_state_init<<<blocks_per_grid, threads_per_block>>>(image_width, image_height, d_random_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // tworzenie świata
    constexpr size_t hittable_list_size = 22*22+1+3;
    Hittable** d_hitlist;
    checkCudaErrors(cudaMalloc(&d_hitlist, sizeof(Hittable*) * hittable_list_size));
    checkCudaErrors(cudaDeviceSynchronize());
    constexpr size_t material_list_size = 22*22+1+3;
    Material** d_material_list;
    checkCudaErrors(cudaMalloc(&d_material_list, sizeof(Material*) * material_list_size));
    checkCudaErrors(cudaDeviceSynchronize());
    Hittable** d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable*)));
    checkCudaErrors(cudaDeviceSynchronize());
    create_world<<<1,1>>>(hittable_list_size, d_hitlist, material_list_size, d_material_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // tworzenie kamery
    Camera* d_cam;
    checkCudaErrors(cudaMalloc(&d_cam, sizeof(Camera)));
    Vector3 look_from = Vector3(13,2,3);
    Vector3 look_at = Vector3(0,1,0);
    int sample_count = 128;
    auto camera = Camera(image_height, aspect_ratio, field_of_view, sample_count, look_from, look_at, d_world, d_random_state);
    checkCudaErrors(cudaMemcpy(d_cam, &camera, sizeof(Camera), cudaMemcpyHostToDevice));


    render<<<blocks_per_grid, threads_per_block>>>(image_width, image_height, d_cam, d_buffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    gamma_2_correction<<<blocks_per_grid, threads_per_block>>>(image_width, image_height, d_buffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // kopiowanie gotowego bufora obrazu z gpu do hosta
    checkCudaErrors(cudaMemcpy(fb_host.data(), d_buffer, sizeof(Vector3) * buffer_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    // zwolnienie pamięci GPU
    destroy_world<<<1,1>>>(hittable_list_size, d_hitlist, material_list_size, d_material_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(d_buffer));
    checkCudaErrors(cudaFree(d_material_list));
    checkCudaErrors(cudaFree(d_hitlist));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_random_state));
    checkCudaErrors(cudaDeviceSynchronize());

    export_framebuffer_to_bitmap(fb_host, image_width, image_height, "image.bmp");

    return 0;
}
