/**
 * @file Camera.cuh
 * @brief Definicja klasy Camera.
 */

#ifndef CAMERA_CUH
#define CAMERA_CUH
#include <curand_kernel.h>

#include "Hittable.cuh"
#include "Vector3.cuh"

/**
 * @class Camera
 * @brief Klasa reprezentująca kamerę w ray tracerze.
 *
 * Klasa ta reprezentuje kamerę w scenie 3D, umożliwia renderowanie pikseli na podstawie pozycji kamery,
 * kierunku patrzenia oraz obiektów w scenie.
 */
class Camera {
public:
    /**
     * @brief Konstruktor klasy Camera.
     *
     * Inicjalizuje kamerę z podanymi parametrami.
     *
     * @param _image_height Wysokość obrazu w pikselach.
     * @param _aspect_ratio Stosunek szerokości do wysokości obrazu.
     * @param _vfov_deg Kąt widzenia w pionie w stopniach.
     * @param _sample_count Liczba próbek na jeden piksel.
     * @param look_from Pozycja kamery w przestrzeni 3D.
     * @param look_at Punkt, na który kamera jest skierowana.
     * @param _world Wskaźnik do tablicy obiektów w scenie.
     * @param _curand_state Wskaźnik do stanu generatora liczb losowych.
     */
    __host__ __device__  Camera(
        int _image_height,
        float _aspect_ratio,
        float _vfov_deg,
        int _sample_count,
        Vector3 look_from,
        Vector3 look_at,
        Vector3 _background_color,
        Hittable** _world,
        curandState* _curand_state
    );

    /**
     * @brief Renderuje piksel na podstawie współrzędnych.
     *
     * @param x Współrzędna x piksela.
     * @param y Współrzędna y piksela.
     *
     * @return Kolor piksela.
     */
    __device__ Vector3 render_pixel(int x, int y);

private:
    /**
     * @brief Oblicza kolor pojedynczego promienia.
     *
     * @param r Promień do obliczenia koloru.
     * @param depth Głębokość rekurencji, tzn. ile razy promień może się odbić od elementów na scenie.
     * @param local_random_state Wskaźnik do lokalnego stanu generatora liczb losowych.
     *
     * @return Wektor koloru dla danego promienia.
     */
    __device__ Vector3 get_ray_color(const Ray& r, int depth, curandState* local_random_state);

    int image_width;                               ///< Szerokość obrazu w pikselach.
    int image_height;                              ///< Wysokość obrazu w pikselach.
    int sample_count;                              ///< Ilość próbek na jeden piksel.
    float aspect_ratio;                            ///< Stosunek szerokości do wysokości obrazu.
    Vector3 camera_center;                         ///< Położenie środka kamery.
    Vector3 pixel_delta_x, pixel_delta_y;          ///< Wektory o długości przestrzeni między środkami pikseli kamery.
    Vector3 viewport_upper_left_pixel_center;      ///< Położenie środka lewego górnego piksela.
    Vector3 background_color;                      ///< Kolor tła (nieba).
    Hittable** world;                              ///< Wskaźnik do tablicy obiektów w scenie.
    curandState* curand_state;                     ///< Wskaźnik do stanu generatora liczb losowych.
};

#endif //CAMERA_CUH
