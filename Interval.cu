/**
* @file Interval.cu
 * @brief Implementacja klasy Interval.
 */

#include "Interval.cuh"

__device__ Interval::Interval() : start(0), end(0) {}

__device__ Interval::Interval(float _start, float _end) : start(_start), end(_end) {}

__device__ float Interval::size() const {
    return end - start;
}

__device__ bool Interval::contains(float value) const {
    return value > start && value < end;
}
