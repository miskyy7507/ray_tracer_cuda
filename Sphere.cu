#include "Sphere.cuh"

__device__ Sphere::Sphere(const Vector3 &_center, float _radius, Material* _material)
    : center(_center), radius(_radius), material(_material) {}

__device__ bool Sphere::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const {
    Vector3 oc = this->center - r.origin(); // co to jest?

    float a = r.direction().dot(r.direction());
    // float b = -2.0f * ray.direction().dot(oc);
    float h = r.direction().dot(oc);
    float c = oc.dot(oc) - this->radius * this->radius;

    // float discriminant = b*b - 4*a*c;
    float discriminant = h*h - a*c;

    if (discriminant < 0.0f) {
        return false;
    }

    float sqrtd = sqrtf(discriminant);

    float root;
    root = (h - sqrtd) / a;
    if (root <= t_min || root >= t_max) {
        root = (h + sqrtd) / a;
        if (root <= t_min || root >= t_max) {
            return false;
        }
    }

    // float root1 = (h - sqrtd) / a;
    // float root2 = (h + sqrtd) / a;
    //
    // bool root1_in_t = root1 > t_min && root1 < t_max;
    // bool root2_in_t = root2 > t_min && root2 < t_max;
    //
    // return !(root1_in_t || root2_in_t);

    rec.point = r.point_at(root);
    rec.normal = (rec.point - this->center) / this->radius;
    rec.material = this->material;
    rec.t = root;

    return true;
}

__global__ void create_sphere(Vector3 center, float radius, int mat_index, Material** mat, Hittable** list, size_t index) {
    list[index] = new Sphere(center, radius, mat[mat_index]);
}