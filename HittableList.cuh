/**
 * @file HittableList.cuh
 * @brief Definicja klasy HittableList.
 */

#ifndef HITTABLELIST_CUH
#define HITTABLELIST_CUH
#include "Hittable.cuh"

/**
 * @class HittableList
 * @brief Klasa reprezentująca listę obiektów Hittable.
 *
 * Klasa HittableList reprezentuje listę obiektów, które mogą być
 * trafione przez promień. Umożliwia wykrywanie kolizji z wieloma
 * obiektami w scenie.
 * Klasa ta przechowuje wskaźniki do obiektów Hittable i implementuje
 * metodę hit, która sprawdza kolizję promienia z każdym obiektem
 * w liście.
 */
class HittableList: public Hittable {
public:
    HittableList() = delete;

    /**
     * @brief Konstruktor klasy HittableList z listą obiektów.
     *
     * @param l Wskaźnik do tablicy wskaźników do obiektów Hittable.
     * @param n Liczba obiektów w liście.
     */
    __device__ HittableList(Hittable** l, const size_t n);

    /**
     * @brief Metoda sprawdzająca kolizję promienia z obiektami w liście.
     *
     * Sprawdza, czy dany promień koliduje z którymkolwiek obiektem
     * w liście. Zapisuje informacje o kolizji w strukturze HitRecord.
     *
     * @param r Promień, który ma być sprawdzony pod kątem kolizji.
     * @param ray_t Zakres wartości t, w którym promień może kolidować.
     * @param rec Struktura, w której zostaną zapisane informacje o kolizji.
     * @return true, jeśli promień koliduje z którymkolwiek obiektem; false w przeciwnym razie.
     */
    __device__ bool hit(const Ray &r, Interval ray_t, HitRecord &rec) const override;

private:
    Hittable** objects; ///< Wskaźnik do tablicy obiektów Hittable.
    size_t length; ///< Liczba obiektów w liście.
};

#endif //HITTABLELIST_CUH
