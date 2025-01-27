#include "HittableList.cuh"

__device__ HittableList::HittableList(Hittable **l, const size_t n): objects(l), length(n) {}

__device__ bool HittableList::hit(const Ray &r, Interval ray_t, HitRecord &rec) const {
    HitRecord temp_rec;
    bool did_hit_anything = false;
    float closest_so_far = ray_t.end;
    for (int i = 0; i < length; i++) {
        if (objects[i]->hit(r, Interval(ray_t.start, closest_so_far), temp_rec)) {
            did_hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return did_hit_anything;
}
