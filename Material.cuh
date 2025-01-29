/**
 * @file Material.cuh
 * @brief Definicja klasy Material.
 */

#ifndef MATERIAL_CUH
#define MATERIAL_CUH
#include "Hittable.cuh"
#include "Ray.cuh"

/**
 * @class Material
 * @brief Klasa bazowa dla materiałów w ray tracingu.
 *
 * Klasa ta definiuje interfejs dla różnych typów materiałów,
 * które mogą być używane w scenie. Zawiera metody do rozpraszania
 * promieni oraz emitowania kolorów, które mogą być nadpisywane
 * przez klasy dziedziczące.
 */
class Material {
public:
    __device__ virtual ~Material() {}

    /**
     * @brief Metoda rozpraszania promienia.
     *
     * @param r_in Promień, który ma być rozpraszany.
     * @param rec Struktura HitRecord zawierająca informacje o kolizji.
     * @param attenuation Wskaźnik do wektora, który będzie zawierał współczynnik tłumienia po rozproszeniu.
     * @param scattered Wskaźnik do promienia, który będzie zawierał nowo rozproszony promień.
     * @param local_random_state Wskaźnik do lokalnego stanu generatora liczb losowych.
     * @return true, jeśli promień został pomyślnie rozproszony, false w przeciwnym razie.
     */
    __device__ virtual bool scatter(
        const Ray& r_in,
        const HitRecord& rec,
        Vector3& attenuation,
        Ray& scattered,
        curandState* local_random_state
    ) const {
        return false;
    }

    /**
     * @brief Metoda zwracająca kolor emitowany przez materiał.
     *
     * @return Wektor koloru emitowanego przez materiał.
     */
    __device__ virtual Vector3 emitted_color() const {
        return Vector3(0.0f, 0.0f, 0.0f);
    }
};

#endif //MATERIAL_CUH
