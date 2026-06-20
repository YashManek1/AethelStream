// galore_apply.cu — S0 stub
#include "galore_apply.cuh"

__global__ void galore_apply_kernel(
    float* param, const float* grad_lr,
    float* m, float* v,
    float clip_scale, float beta1, float beta2, float eps, float lr,
    int n_elements, int step)
{
    // S0 stub: no-op
    (void)param; (void)grad_lr; (void)m; (void)v;
    (void)clip_scale; (void)beta1; (void)beta2; (void)eps; (void)lr;
    (void)n_elements; (void)step;
}

extern "C" void doublepass_galore_apply(
    float* param, const float* grad_lr,
    float* m, float* v,
    float clip_scale, float beta1, float beta2, float eps, float lr,
    int n_elements, int step)
{
    // S0 stub — real kernel launch in Sprint implementing GaLore Adam apply
    (void)param; (void)grad_lr; (void)m; (void)v;
    (void)clip_scale; (void)beta1; (void)beta2; (void)eps; (void)lr;
    (void)n_elements; (void)step;
}
