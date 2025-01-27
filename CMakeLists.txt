cmake_minimum_required(VERSION 3.30)
project(ray_tracer_cuda CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(ray_tracer_cuda main.cu
        Vector3.cu
        Vector3.cuh
        stb_image_write.h
        export_framebuffer_to_bitmap.h
        Ray.cu
        Ray.cuh
        Hittable.cuh
        Sphere.cu
        Sphere.cuh
        HittableList.cu
        HittableList.cuh
        Camera.cu
        Camera.cuh
        rtcuda.h
        Material.cuh
        Matte.cu
        Matte.cuh
        Metal.cu
        Metal.cuh
        Interval.cu
        Interval.cuh)

set_target_properties(ray_tracer_cuda PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
