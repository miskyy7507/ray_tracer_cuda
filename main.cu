#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <curand_kernel.h>
#include "json.hpp"
using json = nlohmann::json;
#include "Camera.cuh"
#include "Vector3.cuh"
#include "export_framebuffer_to_bitmap.h"
#include "Hittable.cuh"
#include "HittableList.cuh"
#include "Quad.cuh"
#include "Matte.cuh"
#include "Metal.cuh"
#include "Light.cuh"
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

/**
 * @brief Kernel do inicjalizacji stanu generatora liczb losowych.
 *
 * Funkcja ta inicjalizuje stan generatora liczb losowych dla każdego
 * piksela w obrazie o zadanej szerokości i wysokości. Używa
 * biblioteki CURAND do generowania losowych liczb.
 *
 * @param width Szerokość obrazu.
 * @param height Wysokość obrazu.
 * @param rand_state Wskaźnik do tablicy stanów generatora liczb losowych.
 *
 * Kernel jest uruchamiany równolegle, a każdy wątek odpowiada
 * za jeden piksel w obrazie. Indeks piksela jest obliczany na
 * podstawie pozycji wątku w siatce.
 */
__global__ void random_state_init(int width, int height, curandState* rand_state) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;
    unsigned int pixel_index = y * width + x;

    curand_init(1+pixel_index, 0, 0, &rand_state[pixel_index]);
}

/**
 * @brief Kernel do tworzenia świata z listy obiektów Hittable.
 *
 * Funkcja ta tworzy obiekt HittableList, który przechowuje wskaźniki
 * do obiektów Hittable. Jest to użyteczne w kontekście ray tracingu,
 * gdzie wiele obiektów musi być zarządzanych jako jedna lista.
 *
 * Funkcja jest uruchamiana tylko przez jeden wątek (wątek o indeksie
 * (0, 0)), aby uniknąć wielokrotnego tworzenia obiektu HittableList.
 *
 * @param hittable_list_size Liczba obiektów Hittable w liście.
 * @param hittable_list Wskaźnik do tablicy wskaźników do obiektów Hittable.
 * @param world Wskaźnik do wskaźnika, w którym zostanie zapisany
 *               wskaźnik do nowo utworzonego obiektu HittableList.
 */
__global__ void create_world(
    const size_t hittable_list_size,
    Hittable** hittable_list,
    Hittable** world
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    *world = new HittableList(hittable_list, hittable_list_size);
}

/**
 * @brief Kernel do usuwania świata z pamięci GPU.
 *
 * Funkcja ta zapewnia prawidłowe zwolnienie pamięci na GPU, usuwając
 * wszystkie obiekty Hittable oraz materiały, które zostały wcześniej
 * utworzone. Jest to ważne dla zarządzania pamięcią i unikania
 * wycieków pamięci w aplikacjach wykorzystujących GPU.
 *
 * Funkcja jest uruchamiana tylko przez jeden wątek (wątek o indeksie
 * (0, 0)), aby uniknąć wielokrotnego usuwania tych samych obiektów.
 *
 * @param hittable_list_size Liczba obiektów Hittable w liście do usunięcia.
 * @param hittable_list Wskaźnik do tablicy wskaźników do obiektów Hittable.
 * @param material_list_size Liczba materiałów w liście do usunięcia.
 * @param material_list Wskaźnik do tablicy wskaźników do materiałów.
 * @param world Wskaźnik do wskaźnika, który wskazuje na obiekt HittableList
 *               do usunięcia.
 */
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

    buffer[pixel_index].x = sqrtf(buffer[pixel_index].x > 1.0f ? 1.0f : buffer[pixel_index].x);
    buffer[pixel_index].y = sqrtf(buffer[pixel_index].y > 1.0f ? 1.0f : buffer[pixel_index].y);
    buffer[pixel_index].z = sqrtf(buffer[pixel_index].z > 1.0f ? 1.0f : buffer[pixel_index].z);
}

/**
 * @brief Walidacja pliku JSON.
 *
 * Funkcja ta sprawdza, czy podany plik JSON jest poprawny
 * pod względem składniowym. Zwraca wartość true, jeśli walidacja
 * przebiegła pomyślnie, w przeciwnym razie zwraca false.
 *
 * @param file Obiekt typu json, który reprezentuje plik JSON do walidacji.
 * @return true Jeśli walidacja pliku JSON przebiegła pomyślnie.
 * @return false Jeśli plik JSON jest niepoprawny lub zawiera błędy.
 */
bool validate_json(const json& file) {
    bool validate_success = true;
    if (!file.contains("camera") || !file["camera"].is_object()) {
        std::cerr << "Missing 'camera' field in file\n";
        validate_success = false;
    } else {
        if (!file["camera"].contains("width") || !file["camera"]["width"].is_number()) {
            std::cerr << "Missing 'width' property in .camera\n";
            validate_success = false;
        }
        if (!file["camera"].contains("height") || !file["camera"]["height"].is_number()) {
            std::cerr << "Missing 'height' property in .camera\n";
            validate_success = false;
        }
        if (!file["camera"].contains("fov") || !file["camera"]["fov"].is_number()) {
            std::cerr << "Missing 'fov' property in .camera\n";
            validate_success = false;
        }
        if (!file["camera"].contains("lookFrom") || !file["camera"]["lookFrom"].is_array() || file["camera"]["lookFrom"].size() != 3) {
            std::cerr << "Missing 'lookFrom' vector in .camera\n";
            validate_success = false;
        }
        if (!file["camera"].contains("lookAt") || !file["camera"]["lookAt"].is_array() || file["camera"]["lookAt"].size() != 3) {
            std::cerr << "Missing 'lookAt' vector in .camera\n";
            validate_success = false;
        }
        if (!file["camera"].contains("backgroundColor") || !file["camera"]["backgroundColor"].is_array() || file["camera"]["backgroundColor"].size() != 3) {
            std::cerr << "Missing 'backgroundColor' vector in .camera\n";
            validate_success = false;
        }
        if (!file["camera"].contains("sampleCount") || !file["camera"]["sampleCount"].is_number()) {
            std::cerr << "Missing 'sampleCount' property in .camera\n";
            validate_success = false;
        }
    }

    size_t material_list_size = 0;

    if (!file.contains("materials") || !file["materials"].is_array()) {
        std::cerr << "Missing 'materials' array in file\n";
        validate_success = false;
    } else {
        material_list_size = file["materials"].size();
        for (const auto& mat : file["materials"]) {
            if (!mat.is_object()) {
                std::cerr << "Bad material type in material array\n";
                validate_success = false;
                continue;
            }
            if (!mat.contains("name") || !mat["name"].is_string()) {
                std::cerr << "Missing 'name' in material entry\n";
                validate_success = false;
                continue;
            }
            if (!mat.contains("data") || !mat["data"].is_object()) {
                std::cerr << "Missing 'data' in material entry\n";
                validate_success = false;
                continue;
            }

            if (mat["name"] == "matte") {
                if (!mat["data"].contains("color") || !mat["data"]["color"].is_array() || mat["data"]["color"].size() != 3) {
                    std::cerr << "Missing 'color' vector in matte material data\n";
                    validate_success = false;
                }
            } else if (mat["name"] == "metal") {
                if (!mat["data"].contains("color") || !mat["data"]["color"].is_array() || mat["data"]["color"].size() != 3) {
                    std::cerr << "Missing 'color' vector in metal material data\n";
                    validate_success = false;
                }
                if (!mat["data"].contains("roughness") || !mat["data"]["roughness"].is_number()) {
                    std::cerr << "Missing 'roughness' in metal material data\n";
                    validate_success = false;
                }
            } else if (mat["name"] == "light") {
                if (!mat["data"].contains("color") || !mat["data"]["color"].is_array() || mat["data"]["color"].size() != 3) {
                    std::cerr << "Missing 'color' vector in light material data\n";
                    validate_success = false;
                }
            } else {
                std::cerr << "Unknown material: " << mat["name"] << std::endl;
                validate_success = false;
            }
        }
    }
    if (!file.contains("hittables") || !file["hittables"].is_array()) {
        std::cerr << "Missing 'hittables' array in file\n";
        validate_success = false;
    } else {
        for (const auto& hittable : file["hittables"]) {
            if (!hittable.is_object()) {
                std::cerr << "Bad hittable type in hittable array\n";
                validate_success = false;
                continue;
            }
            if (!hittable.contains("name") || !hittable["name"].is_string()) {
                std::cerr << "Missing 'name' in hittable entry\n";
                validate_success = false;
                continue;
            }
            if (!hittable.contains("data") || !hittable["data"].is_object()) {
                std::cerr << "Missing 'data' in material entry\n";
                validate_success = false;
                continue;
            }

            if (!hittable["data"].contains("materialIndex") || !hittable["data"]["materialIndex"].is_number()) {
                std::cerr << "Missing 'materialIndex' in hittable data\n";
                validate_success = false;
            } else {
                size_t hittable_index = hittable["data"]["materialIndex"].get<size_t>();
                if (hittable_index >= material_list_size) {
                    std::cerr << "Material index out of range\n";
                    validate_success = false;
                }
            }

            if (hittable["name"] == "sphere") {
                if (!hittable["data"].contains("center") || !hittable["data"]["center"].is_array() || hittable["data"]["center"].size() != 3) {
                    std::cerr << "Missing 'center' vector in sphere hittable data\n";
                    validate_success = false;
                }
                if (!hittable["data"].contains("radius") || !hittable["data"]["radius"].is_number()) {
                    std::cerr << "Missing 'radius' in sphere hittable data\n";
                    validate_success = false;
                }
            } else if (hittable["name"] == "quad") {
                if (!hittable["data"].contains("q") || !hittable["data"]["q"].is_array() || hittable["data"]["q"].size() != 3) {
                    std::cerr << "Missing 'q' vector in quad hittable data\n";
                    validate_success = false;
                }
                if (!hittable["data"].contains("u") || !hittable["data"]["u"].is_array() || hittable["data"]["u"].size() != 3) {
                    std::cerr << "Missing 'u' vector in quad hittable data\n";
                    validate_success = false;
                }
                if (!hittable["data"].contains("v") || !hittable["data"]["v"].is_array() || hittable["data"]["v"].size() != 3) {
                    std::cerr << "Missing 'v' vector in quad hittable data\n";
                    validate_success = false;
                }
            } else {
                std::cerr << "Unknown hittable shape: " << hittable["name"] << std::endl;
                validate_success = false;
            }
        }
    }

    return validate_success;
}

/**
 * @brief Ładowanie materiałów z pliku JSON do pamięci GPU.
 *
 * Funkcja alokuje pamięć na liście materiałów w pamięci GPU, a następnie
 * iteruje przez każdy materiał w pliku JSON, tworząc odpowiednie obiekty
 * materiałów na GPU. Po zakończeniu ładowania, synchronizuje urządzenie
 * GPU, aby upewnić się, że wszystkie operacje zostały zakończone.
 *
 * @param file Obiekt typu json, który reprezentuje plik JSON zawierający
 *              definicje materiałów.
 * @param d_material_list Pusty wskaźnik, który będzie wskazywał
 *                        na tablicę wskaźników do obiektów materiałów
 *                        w pamięci GPU.
 */
void load_materials(const json& file, Material**& d_material_list) {
    size_t d_material_list_size = file["materials"].size();
    cudaMalloc(&d_material_list, sizeof(Material*) * d_material_list_size);

    size_t current_index = 0;
    for (const auto& mat : file["materials"]) {
        if (mat["name"] == "matte") {
            const auto color = Vector3(mat["data"]["color"][0].get<float>(), mat["data"]["color"][1].get<float>(), mat["data"]["color"][2].get<float>());
            create_matte<<<1,1>>>(color, d_material_list, current_index);
            checkCudaErrors(cudaGetLastError());
        } else if (mat["name"] == "metal") {
            const auto color = Vector3(mat["data"]["color"][0].get<float>(), mat["data"]["color"][1].get<float>(), mat["data"]["color"][2].get<float>());
            const float roughness = mat["data"]["roughness"].get<float>();
            create_metal<<<1,1>>>(color, roughness, d_material_list, current_index);
            checkCudaErrors(cudaGetLastError());
        } else if (mat["name"] == "light") {
            const auto color = Vector3(mat["data"]["color"][0].get<float>(), mat["data"]["color"][1].get<float>(), mat["data"]["color"][2].get<float>());
            create_light<<<1,1>>>(color, d_material_list, current_index);
            checkCudaErrors(cudaGetLastError());
        } else {
            std::cerr << "Unknown material: " << mat["name"] << std::endl;
            assert(false);
        }
        current_index++;
    }
    checkCudaErrors(cudaDeviceSynchronize());
}

/**
 * @brief Ładowanie obiektów z pliku JSON do pamięci GPU.
 *
 * Funkcja alokuje pamięć na liście obiektów w pamięci GPU, a następnie
 * iteruje przez każdy obiekt w pliku JSON, tworząc odpowiednie obiekty na GPU.
 *
 * @param file Plik JSON zawierający definicje materiałów.
 * @param d_material_list Pusty wskaźnik, który będzie wskazywał
 *                        na tablicę wskaźników do obiektów w pamięci GPU.
 * @return Rozmiar tablicy d_material_list.
 */
int load_hittables(const json& file, Hittable**& d_hittable_list, Material**& d_material_list) {
    size_t d_hittable_list_size = file["hittables"].size();
    cudaMalloc(&d_hittable_list, sizeof(Hittable*) * d_hittable_list_size);

    size_t current_index = 0;
    for (const auto& hittable : file["hittables"]) {
        if (hittable["name"] == "sphere") {
            const auto center = Vector3(hittable["data"]["center"][0].get<float>(), hittable["data"]["center"][1].get<float>(), hittable["data"]["center"][2].get<float>());
            const float radius = hittable["data"]["radius"].get<float>();
            const int material_index = hittable["data"]["materialIndex"].get<int>();
            create_sphere<<<1,1>>>(center, radius, material_index, d_material_list, d_hittable_list, current_index);
            checkCudaErrors(cudaGetLastError());
        } else if (hittable["name"] == "quad") {
            const auto Q = Vector3(hittable["data"]["q"][0].get<float>(), hittable["data"]["q"][1].get<float>(), hittable["data"]["q"][2].get<float>());
            const auto u = Vector3(hittable["data"]["u"][0].get<float>(), hittable["data"]["u"][1].get<float>(), hittable["data"]["u"][2].get<float>());
            const auto v = Vector3(hittable["data"]["v"][0].get<float>(), hittable["data"]["v"][1].get<float>(), hittable["data"]["v"][2].get<float>());
            const int material_index = hittable["data"]["materialIndex"].get<int>();
            create_quad<<<1,1>>>(Q, u, v, material_index, d_material_list, d_hittable_list, current_index);
        } else {
            std::cerr << "Unknown material: " << hittable["name"] << std::endl;
            assert(false);
        }
        current_index++;
    }
    checkCudaErrors(cudaDeviceSynchronize());

    return d_hittable_list_size;
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Użycie: " << argv[0] << " <plik_konfiguracyjny> <plik_wyjściowy>" << std::endl;
        exit(1);
    }

    std::string config_file_path = argv[1];
    std::string output_file_path = argv[2];


    std::cerr << "Plik konfiguracyjny: " << config_file_path << "\n";
    json json_file;

    std::ifstream json_file_stream(config_file_path);
    if (!json_file_stream.is_open()) {
        std::cerr << "Nie można było otworzyć pliku konfiguracyjnego. Wychodzę.\n";
        json_file_stream.close();
        return 1;
    }
    json_file_stream >> json_file;
    json_file_stream.close();

    if (!validate_json(json_file)) {
        std::cout << "Błąd walidacji pliku konfiguracyjnego. Wychodzę.\n";
        return 1;
    }

    auto camera_settings = json_file["camera"];

    int image_width = camera_settings["width"];
    int image_height = camera_settings["height"];
    float aspect_ratio = (float)image_width / image_height;
    float field_of_view = camera_settings["fov"]; // pole widzenia wertykalne w kątach
    Vector3 look_from = Vector3(camera_settings["lookFrom"][0], camera_settings["lookFrom"][1], camera_settings["lookFrom"][2]);
    Vector3 look_at = Vector3(camera_settings["lookAt"][0], camera_settings["lookAt"][1], camera_settings["lookAt"][2]);
    Vector3 background_color = Vector3(camera_settings["backgroundColor"][0], camera_settings["backgroundColor"][1], camera_settings["backgroundColor"][2]);
    int sample_count = camera_settings["sampleCount"];

    int buffer_size = image_height * image_width;

    auto fb_host = std::vector<Vector3>(buffer_size);

    constexpr int block_x_size = 8;
    constexpr int block_y_size = 8;
    dim3 threads_per_block(block_x_size,block_y_size);
    dim3 blocks_per_grid(
        static_cast<int>(std::ceil(static_cast<float>(image_width) / block_x_size)),
        static_cast<int>(std::ceil(static_cast<float>(image_height) / block_y_size))
    );

    // alokacja pamięci gpu

    // stan cuRAND (generator liczb pseudolosowych)
    curandState *d_random_state;
    checkCudaErrors(cudaMalloc(&d_random_state, buffer_size * sizeof(curandState)));
    checkCudaErrors(cudaDeviceSynchronize());
    random_state_init<<<blocks_per_grid, threads_per_block>>>(image_width, image_height, d_random_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Vector3* d_buffer;
    checkCudaErrors(cudaMalloc(&d_buffer, sizeof(Vector3) * buffer_size));

    // tworzenie świata
    Material** d_material_list;
    load_materials(json_file, d_material_list);

    Hittable** d_hitlist;
    int hittable_list_size = load_hittables(json_file, d_hitlist, d_material_list);

    Hittable** d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(Hittable*)));
    checkCudaErrors(cudaDeviceSynchronize());
    create_world<<<1,1>>>(hittable_list_size, d_hitlist, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // tworzenie kamery
    Camera* d_cam;
    checkCudaErrors(cudaMalloc(&d_cam, sizeof(Camera)));
    auto camera = Camera(image_height, aspect_ratio, field_of_view, sample_count, look_from, look_at,background_color, d_world, d_random_state);
    checkCudaErrors(cudaMemcpy(d_cam, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    std::clog << "Renderowanie... \n";
    auto render_begin = std::chrono::steady_clock::now();
    render<<<blocks_per_grid, threads_per_block>>>(image_width, image_height, d_cam, d_buffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    auto render_end = std::chrono::steady_clock::now();
    std::clog << "Wyrenderowano w " << std::chrono::duration_cast<std::chrono::milliseconds>(render_end - render_begin).count() / 1000.f << "s." << std::endl;

    gamma_2_correction<<<blocks_per_grid, threads_per_block>>>(image_width, image_height, d_buffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // kopiowanie gotowego bufora obrazu z gpu do hosta
    checkCudaErrors(cudaMemcpy(fb_host.data(), d_buffer, sizeof(Vector3) * buffer_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    // zwolnienie pamięci GPU
    destroy_world<<<1,1>>>(hittable_list_size, d_hitlist, 4, d_material_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(d_buffer));
    checkCudaErrors(cudaFree(d_material_list));
    checkCudaErrors(cudaFree(d_hitlist));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_random_state));

    export_framebuffer_to_bitmap(fb_host, image_width, image_height, output_file_path.c_str());

    return 0;
}
