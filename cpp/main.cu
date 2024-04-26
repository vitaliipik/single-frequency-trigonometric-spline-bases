#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel to calculate the i-th B-spline basis function of order k at parameter t
__device__ float basis_function(int i, int k, float t, float* knots) {
    if (k == 1) {
        return (knots[i] <= t && t < knots[i + 1]) ? 1.0f : 0.0f;
    }

    float denom1 = knots[i + k - 1] - knots[i];
    float denom2 = knots[i + k] - knots[i + 1];

    float coeff1 = (denom1 == 0.0f) ? 0.0f : sinf((t - knots[i]) / 2.0f) / sinf(denom1 / 2.0f);
    float coeff2 = (denom2 == 0.0f) ? 0.0f : sinf((knots[i + k] - t) / 2.0f) / sinf((knots[i + k] - knots[i + 1]) / 2.0f);

    return coeff1 * basis_function(i, k - 1, t, knots) + coeff2 * basis_function(i + 1, k - 1, t, knots);
}

// CUDA kernel to calculate the B-spline curve at parameter t
__global__ void b_spline_curve_kernel(float* control_points, int num_control_points, int degree, float* knots, int num_points, float* curve_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        float t = (3.0f / (num_points - 1)) * idx;
        float result_x = 0.0f;
        float result_y = 0.0f;

        for (int i = 0; i <= num_control_points; ++i) {
            float basis = basis_function(i, degree + 1, t, knots);
            result_x += control_points[2 * i] * basis;
            result_y += control_points[2 * i + 1] * basis;
        }

        curve_points[2 * idx] = result_x;
        curve_points[2 * idx + 1] = result_y;
    }
}

// Host function to compute B-spline curve using CUDA
void compute_b_spline_curve(float* h_control_points, int num_control_points, int degree, float* h_knots, int num_points, float* h_curve_points) {
    float *d_control_points, *d_knots, *d_curve_points;
    cudaMalloc((void**)&d_control_points, 2 * num_control_points * sizeof(float));
    cudaMalloc((void**)&d_knots, (num_control_points + degree + 1) * sizeof(float));
    cudaMalloc((void**)&d_curve_points, 2 * num_points * sizeof(float));

    cudaMemcpy(d_control_points, h_control_points, 2 * num_control_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_knots, h_knots, (num_control_points + degree + 1) * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;

    b_spline_curve_kernel<<<numBlocks, blockSize>>>(d_control_points, num_control_points, degree, d_knots, num_points, d_curve_points);

    cudaMemcpy(h_curve_points, d_curve_points, 2 * num_points * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_control_points);
    cudaFree(d_knots);
    cudaFree(d_curve_points);
}

int main() {
    // Example usage:
    int num_control_points = 6;
    int degree = 5;
    float control_points[] = {0, 0, 1, 3, 2, 4, 3, 4, 4, 0, 6, 2};
    float knots[] = {0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6};  // Example knot vector
    int num_points = 1000;
    float* curve_points = new float[2 * num_points];

    compute_b_spline_curve(control_points, num_control_points, degree, knots, num_points, curve_points);

    // Printing or plotting the curve points
    for (int i = 0; i < num_points; ++i) {
        std::cout << curve_points[2 * i] << ", " << curve_points[2 * i + 1] << std::endl;
    }

    delete[] curve_points;
    return 0;
}
