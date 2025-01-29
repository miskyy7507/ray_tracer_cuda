/**
 * @file Quad.cu
 * @brief Implementacja klasy Quad.
 */

#include "Quad.cuh"

__device__ Quad::Quad(const Vector3 &Q, const Vector3 &_u, const Vector3 &_v, Material *_material)
    : Q(Q), u(_u), v(_v), material(_material) {
    auto n = u.cross(v);
    normal = n.normalized(); // normalna równoległoboka według zasady prawej ręki
    D = normal.dot(Q);
    w = n / n.dot(n);
}

__device__  bool Quad::hit(const Ray &r, Interval ray_t, HitRecord &rec) const {
    // Jeżeli promień jest równeległy do równoległoboka, nie ma kolizji.
    float denom = normal.dot(r.direction());
    if (denom == 0) {
        return false;
    }

    // Obliczamy t kiedy promień trafił na płaszczyznę.
    float t = (D - normal.dot(r.origin())) / denom;
    if (!ray_t.contains(t)) {
        return false;
    }

    auto intersection = r.point_at(t); // punkt przecięcia się promienia z płaszczyzną

    // W tym momencie wiemy że promień dotknął płaszczyznę na której jest zdefiniowany równoległobok.
    Vector3 planar_hitpoint = intersection - Q;
    float alpha = w.dot(planar_hitpoint.cross(v));
    float beta  = w.dot(u.cross(planar_hitpoint));

    if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) {
        return false;
    }

    rec.point = intersection;
    rec.normal = normal;
    rec.material = material;
    rec.t = t;


    return true;
}

__global__ void create_quad(Vector3 Q, Vector3 u, Vector3 v, int mat_index, Material** mat_list, Hittable** list, size_t index) {
    list[index] = new Quad(Q, u, v, mat_list[mat_index]);
}