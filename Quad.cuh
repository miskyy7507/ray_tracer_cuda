/**
 * @file Quad.cuh
 * @brief Definicja klasy Quad reprezentującej równoległobok w przestrzeni 3D.
 */

#ifndef QUAD_CUH
#define QUAD_CUH
#include "Hittable.cuh"
#include "Vector3.cuh"

/**
 * @class Quad
 * @brief Klasa reprezentująca czworokąt w przestrzeni 3D.
 *
 * Klasa ta dziedziczy po klasie Hittable i implementuje metody do wykrywania kolizji
 * promienia z czworokątem. Przechowuje informacje o wierzchołkach czworokąta, jego materiale oraz
 * normalnej. Umożliwia wykrywanie kolizji promienia z czworokątem.
 */
class Quad: public Hittable {
public:
    /**
     * @brief Konstruktor klasy Quad.
     *
     * @param Q Wierzchołek czworokąta.
     * @param _u Wektor określający jeden z boków czworokąta.
     * @param _v Wektor określający drugi bok czworokąta.
     * @param _material Wskaźnik do materiału czworokąta.
     */
    __device__ Quad(const Vector3& Q, const Vector3& _u, const Vector3& _v, Material* _material);

    /**
     * @brief Metoda sprawdzająca kolizję promienia z czworokątem.
     *
     * @param r Promień, który ma być sprawdzony pod kątem kolizji.
     * @param ray_t Zakres wartości t, w którym promień może kolidować.
     * @param rec Struktura, w której zostaną zapisane informacje o kolizji.
     * @return true Jeśli promień koliduje z czworokątem.
     * @return false Jeśli promień nie koliduje z czworokątem.
     */
    __device__ bool hit(const Ray &r, Interval ray_t, HitRecord &rec) const override;

private:
    Vector3 Q; ///< Wierzchołek czworokąta.
    Vector3 u; ///< Wektor określający jeden z boków czworokąta.
    Vector3 v; ///< Wektor określający drugi bok czworokąta.
    Vector3 w;
    Material* material; ///< Wskaźnik do materiału czworokąta.
    Vector3 normal; ///< Normalna czworokąta.
    float D; ///< Wartość D w równaniu płaszczyzny.
};

/**
 * @brief Kernel do tworzenia czworokąta w pamięci GPU.
 *
 * @param Q Wierzchołek czworokąta.
 * @param u Wektor określający jeden z boków czworokąta.
 * @param v Wektor określający drugi bok czworokąta.
 * @param mat_index Indeks materiału w liście materiałów.
 * @param mat_list Lista wskaźników do materiałów.
 * @param list Lista wskaźników do obiektów Hittable,
 *        w którym zostanie zapisany wskaźnik do tego utworzonego równoległoboku.
 * @param index Indeks, pod którym czworokąt ma być zapisany w liście.
 */
__global__ void create_quad(Vector3 Q, Vector3 u, Vector3 v, int mat_index, Material** mat_list, Hittable** list, size_t index);

#endif //QUAD_CUH
