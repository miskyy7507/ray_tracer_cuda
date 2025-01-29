/**
 * @file Hittable.cuh
 * @brief Definicja klas HitRecord i Hittable.
 *
 * Plik ten zawiera definicje klas używanych do wykrywania kolizji
 * w kontekście ray tracingu. Klasa HitRecord przechowuje informacje
 * o kolizji, natomiast klasa Hittable jest interfejsem dla obiektów,
 * które mogą być trafione przez promień.
 */

#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "Interval.cuh"
#include "Ray.cuh"
#include "Vector3.cuh"

class Material;

/**
 * @class HitRecord
 * @brief Struktura przechowująca informacje o kolizji promienia.
 *
 * Klasa ta zawiera dane dotyczące punktu kolizji, normalnej
 * w punkcie kolizji, materiału obiektu oraz wartości t, która
 * reprezentuje odległość od początku promienia do punktu kolizji.
 */
class HitRecord {
public:
    Vector3 point; ///< Punkt kolizji.
    Vector3 normal; ///< Normalna w punkcie kolizji.
    Material* material; ///< Wskaźnik do materiału obiektu.
    float t; ///< Odległość od początku promienia do punktu kolizji.
};

/**
 * @class Hittable
 * @brief Interfejs dla obiektów, które mogą być trafione przez promień.
 *
 * Klasa ta definiuje metodę hit, która musi być zaimplementowana
 * przez wszystkie klasy dziedziczące. Umożliwia to wykrywanie kolizji
 * promienia z różnymi obiektami w scenie.
 */
class Hittable {
public:
    /**
     * @brief Wirtualny destruktor klasy Hittable.
     *
     * Umożliwia poprawne usunięcie obiektów dziedziczących
     * z tej klasy.
     */
    __device__ virtual ~Hittable() {}

    /**
     * @brief Metoda sprawdzająca kolizję promienia z obiektem.
     *
     * @param r Promień, który ma być sprawdzony pod kątem kolizji.
     * @param ray_t Zakres wartości t, w którym promień może kolidować.
     * @param rec Struktura, w której zostaną zapisane informacje o kolizji.
     * @return true, jeśli promień koliduje z obiektem; false w przeciwnym razie.
     */
    __device__ virtual bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const = 0;
};

#endif //HITTABLE_CUH
