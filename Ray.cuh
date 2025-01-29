/**
 * @file Ray.cuh
 * @brief Definicja klasy Ray.
 */

#ifndef RAY_CUH
#define RAY_CUH
#include "Vector3.cuh"

/**
 * @brief Klasa reprezentująca promień w ray tracerze.
 *
 * Klasa ta przechowuje informacje o położeniu początkowym promienia oraz jego kierunku.
 */
class Ray {
public:
    /**
     * @brief Konstruktor domyślny.
     *
     * Inicjalizuje promień z początkowym położeniem i kierunkiem równym zerowemu wektorowi.
     */
    __device__ Ray();

    /**
     * @brief Konstruktor z parametrami.
     *
     * Inicjalizuje promień z podanym położeniem początkowym i kierunkiem.
     *
     * @param _orig Położenie początkowe promienia.
     * @param _dir Kierunek promienia.
     */
    __device__ Ray(const Vector3& _orig, const Vector3& _dir);

    /**
     * @brief Zwraca położenie początkowe promienia.
     *
     * @return Referencja do wektora reprezentującego położenie początkowe promienia.
     */
    __device__ const Vector3& origin() const;

    /**
     * @brief Zwraca kierunek promienia.
     *
     * @return Referencja do wektora reprezentującego kierunek promienia.
     */
    __device__ const Vector3& direction() const;

    /**
     * @brief Zwraca punkt na promieniu w odległości t od położenia początkowego.
     *
     * @param t Odległość od położenia początkowego wzdłuż kierunku promienia.
     * @return Wektor reprezentujący punkt na promieniu.
     */
    __device__ Vector3 point_at(float t) const;

private:
    Vector3 orig; ///< Położenie początkowe promienia.
    Vector3 dir;  ///< Kierunek promienia.
};

#endif //RAY_CUH
