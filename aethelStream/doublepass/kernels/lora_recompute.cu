// lora_recompute.cu — S0 stub
#include "lora_recompute.cuh"

__global__ void lora_recompute_h_kernel(
    const float* x, const float* a_mat, float* h_out,
    int batch_seq, int d_in, int rank)
{
    // S0 stub: no-op
    (void)x; (void)a_mat; (void)h_out;
    (void)batch_seq; (void)d_in; (void)rank;
}

extern "C" void doublepass_lora_recompute_h(
    const float* x, const float* a_mat, float* h_out,
    int batch_seq, int d_in, int rank)
{
    // S0 stub — real kernel launch in Sprint implementing LoRA recompute
    (void)x; (void)a_mat; (void)h_out;
    (void)batch_seq; (void)d_in; (void)rank;
}
