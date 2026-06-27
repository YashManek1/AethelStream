// quantize_state.cu — absmax INT8 quantisation kernels for AdamW m/v states
#include "quantize_state.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__global__ void find_absmax_f32_kernel(
    const float* __restrict__ src,
    int                        n,
    float* __restrict__        absmax_out
) {
    extern __shared__ float smax[];
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = src[i];
        float a = (v < 0.0f) ? -v : v;
        if (a > local) local = a;
    }
    smax[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smax[threadIdx.x + stride] > smax[threadIdx.x]) {
                smax[threadIdx.x] = smax[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *absmax_out = smax[0];
    }
}

__global__ void quantize_absmax_kernel(
    const float* __restrict__ src,
    int8_t* __restrict__      dst,
    float                     scale,
    int                       n
) {
    const float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float qf = src[i] * inv;
        int   qi = __float2int_rn(qf);
        if (qi >  127) qi =  127;
        if (qi < -127) qi = -127;
        dst[i] = (int8_t)qi;
    }
}

__global__ void dequantize_i8_kernel(
    const int8_t* __restrict__ src,
    float* __restrict__        dst,
    float                      scale,
    int                        n
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = (float)src[i] * scale;
    }
}

extern "C" int galore_quantize_absmax_f32_to_i8(
    const float* src,
    int8_t*      dst,
    float*       scale_out,
    int          n_elements,
    cudaStream_t stream
) {
    if (n_elements <= 0) return -1;

    find_absmax_f32_kernel<<<1, 256, 256 * sizeof(float), stream>>>(src, n_elements, scale_out);
    cudaStreamSynchronize(stream);

    float h_scale = 0.0f;
    cudaMemcpyAsync(&h_scale, scale_out, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (h_scale <= 0.0f) h_scale = 1.0f;
    float quant_scale = h_scale / 127.0f;
    cudaMemcpyAsync(scale_out, &quant_scale, sizeof(float), cudaMemcpyHostToDevice, stream);

    const int block = 256;
    const int grid  = (n_elements + block - 1) / block;
    quantize_absmax_kernel<<<grid, block, 0, stream>>>(src, dst, quant_scale, n_elements);
    return 0;
}

extern "C" int galore_dequantize_i8_to_f32(
    const int8_t* src,
    float*        dst,
    float         scale,
    int           n_elements,
    cudaStream_t stream
) {
    if (n_elements <= 0) return -1;
    const int block = 256;
    const int grid  = (n_elements + block - 1) / block;
    dequantize_i8_kernel<<<grid, block, 0, stream>>>(src, dst, scale, n_elements);
    return 0;
}
