/**
 * @file Vector3.cuh
 * @brief Definicja klasy Vector3.
 */

#ifndef VECTOR3_CUH
#define VECTOR3_CUH

#include <curand_kernel.h>

/**
 * @class Vector3
 * @brief Klasa reprezentująca wektor w trójwymiarowej przestrzeni.
 *
 * Klasa Vector3 reprezentuje wektor w trójwymiarowej przestrzeni.
 * Zawiera metody do wykonywania operacji wektorowych, takich jak
 * dodawanie, odejmowanie, iloczyn skalarny, iloczyn wektorowy oraz
 * normalizacja. Klasa obsługuje również generowanie losowych wektorów.
 * Klasa ta przechowuje współrzędne wektora (x, y, z).
 */
class Vector3 {
public:
    float x, y, z; ///< Współrzędne wektora.

    /**
     * @brief Konstruktor "pustego" wektora (0.0, 0.0, 0.0).
     */
    __device__ __host__ Vector3();

    /**
     * @brief Konstruktor wektora z podanymi współrzędnymi.
     *
     * @param _x Współrzędna x wektora.
     * @param _y Współrzędna y wektora.
     * @param _z Współrzędna z wektora.
     */
    __device__ __host__ Vector3(float _x, float _y, float _z);

    /**
     * @brief Konstruktor wektora z tablicy.
     *
     * @param v Tablica zawierająca współrzędne wektora.
     */
    __device__ __host__ Vector3(float v[]);

    /**
     * @brief Operator negacji wektora.
     *
     * @return Wektor przeciwny.
     */
    __device__ __host__ Vector3 operator-() const;

    /**
     * @brief Operator dodawania wektorów.
     *
     * @param v Wektor, który ma być dodany.
     * @return Wektor będący sumą dwóch wektorów.
     */
    __device__ __host__ Vector3 operator+(const Vector3& v) const;

    /**
     * @brief Operator odejmowania wektorów.
     *
     * @param v Wektor, który ma być odjęty.
     * @return Wektor będący różnicą dwóch wektorów.
     */
    __device__ __host__ Vector3 operator-(const Vector3& v) const;

    /**
     * @brief Operator iloczynu elementarnego (Hadamarda) wektorów.
     *
     * @param v Wektor, z którym ma być wykonany iloczyn.
     * @return Wektor będący iloczynem elementarnym dwóch wektorów.
     */
    __device__ __host__ Vector3 operator*(const Vector3& v) const;

    /**
     * @brief Operator mnożenia wektora przez skalar.
     *
     * @param n Skalar, przez który ma być pomnożony wektor.
     * @return Wektor będący wynikiem mnożenia.
     */
    __device__ __host__ Vector3 operator*(const float& n) const;

    /**
     * @brief Operator dzielenia wektora przez skalar.
     *
     * @param n Skalar, przez który ma być podzielony wektor.
     * @return Wektor będący wynikiem dzielenia.
     */
    __device__ __host__ Vector3 operator/(const float& n) const;

    /**
     * @brief Iloczyn skalarny dwóch wektorów.
     *
     * @param v Wektor, z którym ma być wykonany iloczyn skalarny.
     * @return Wartość iloczynu skalarnego.
     */
    __device__ __host__ float dot(const Vector3& v) const;

    /**
     * @brief Iloczyn wektorowy dwóch wektorów.
     *
     * @param v Wektor, z którym ma być wykonany iloczyn wektorowy.
     * @return Wektor będący wynikiem iloczynu wektorowego.
     */
    __device__ __host__ Vector3 cross(const Vector3& v) const;

    /**
     * @brief Długość wektora do kwadratu.
     *
     *
     * @return Długość wektora do kwadratu.
     */
    __device__ __host__ float length_squared() const;

    /**
     * @brief Długość wektora.
     *
     * @return Długość wektora.
     */
    __device__ __host__ float length() const;

    /**
     * @brief Znormalizowany wektor (o długości 1).
     *
     * @return Wektor znormalizowany.
     */
    __device__ __host__ Vector3 normalized() const;

    /**
     * @brief Generuje wektor o losowych współrzędnych x, y, z w przedziale (0.0, 1.0).
     *
     * @param local_random_state Wskaźnik do lokalnego stanu generatora liczb losowych.
     * @return Wektor z losowymi współrzędnymi.
     */
    __device__ static Vector3 random(curandState* local_random_state);

    /**
     * @brief Generuje wektor o losowych współrzędnych x, y, z w przedziale (min, max).
     *
     * @param min Minimalna wartość dla współrzędnych.
     * @param max Maksymalna wartość dla współrzędnych.
     * @param local_random_state Wskaźnik do lokalnego stanu generatora liczb losowych.
     * @return Wektor z losowymi współrzędnymi w zadanym przedziale.
     */
    __device__ static Vector3 random(float min, float max, curandState* local_random_state);

    /**
     * @brief Generuje wektor o znormalizowanej długości w losowym kierunku.
     *
     * @param local_random_state Wskaźnik do lokalnego stanu generatora liczb losowych.
     * @return Wektor na powierzchni sfery o promieniu 1.
     */
    __device__ static Vector3 random_unit_vector_old(curandState* local_random_state);

    /**
     * @brief Generuje wektor o znormalizowanej długości w losowym kierunku.
     *
     * @param local_random_state Wskaźnik do lokalnego stanu generatora liczb losowych.
     * @return Wektor na powierzchni sfery o promieniu 1.
     */
    __device__ static Vector3 random_unit_vector(curandState* local_random_state);

    /**
     * @brief Odbicie wektora względem danej normalnej.
     *
     * @param normal Normalna, względem której ma być wykonane odbicie.
     * @return Wektor będący wynikiem odbicia.
     */
    __device__ Vector3 reflect(const Vector3& normal) const;
};

#endif //VECTOR3_CUH