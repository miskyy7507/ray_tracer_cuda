#ifndef RTCUDA_H
#define RTCUDA_H

#include <limits>

namespace RTCuda {
    constexpr float INF = std::numeric_limits<float>::infinity();
    constexpr float PI = 3.141592653;

    // // Zwraca liczbę zmiennoprzecinkową w przedziale (0.0f, 1.0f].
    // __device__ inline float random_float(curandState* local_random_state) {
    //     return curand_uniform(local_random_state);
    // }
}

#endif //RTCUDA_H
