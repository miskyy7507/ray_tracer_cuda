/**
* @file Interval.cuh
 * @brief Definicja klasy Interval reprezentującej przedział otwarty.
 */

#ifndef INTERVAL_CUH
#define INTERVAL_CUH

/**
 * @class Interval
 * @brief Klasa reprezentująca przedział otwarty.
 *
 * Klasa ta umożliwia tworzenie przedziałów oraz sprawdzanie, czy dany
 * element należy do tego przedziału. Przedział jest definiowany przez
 * dwa punkty: początek i koniec.
 *
 * Klasa do użytku wyłącznie w kodzie urządzenia GPU (__device__).
 */
class Interval {
public:
    /**
     * @brief Konstruktor domyślny.
     *
     * Inicjalizuje przedział niezdefiniowany, z wartościami `start`
     * i `end` równymi 0.
     */
    __device__ Interval();

    /**
     * @brief Konstruktor z parametrami.
     *
     * Inicjalizuje przedział z podanymi wartościami start i end.
     *
     * @param _start Wartość początkowa przedziału.
     * @param _end Wartość końcowa przedziału.
     */
    __device__ Interval(float _start, float _end);

    /**
     * @brief Oblicza rozmiar przedziału.
     *
     * @return Rozmiar przedziału (end - start).
     */
    __device__ float size() const;

    /**
     * @brief Sprawdza, czy wartość należy do tego przedziału otwartego.
     *
     * @param value Wartość do sprawdzenia.
     * @return true, jeśli wartość należy do przedziału; false w przeciwnym razie.
     */
    __device__ bool contains(float value) const;

    float start; ///< Wartość początkowa przedziału.
    float end;   ///< Wartość końcowa przedziału.
};

#endif //INTERVAL_CUH
