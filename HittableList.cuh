#ifndef HITTABLELIST_CUH
#define HITTABLELIST_CUH
#include "Hittable.cuh"

class HittableList: public Hittable {
public:
    HittableList() = delete;

    __device__ HittableList(Hittable** l, const size_t n);

    __device__ bool hit(const Ray &r, Interval ray_t, HitRecord &rec) const override;

private:
    Hittable** objects;
    size_t length;
};

#endif //HITTABLELIST_CUH
