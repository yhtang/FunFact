extern "C" {
    __global__ void adam(
        float * __restrict w,
        float const * __restrict dw,
        float * __restrict m,
        float * __restrict v,
        const float lr,
        const float beta1,
        const float beta2,
        const float epsilon,
        const int n
    ){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            m[i] = beta1 * m[i] + (1 - beta1) * dw[i];
            v[i] = beta2 * v[i] + (1 - beta2) * dw[i] * dw[i];
            auto mhat = m[i] / (1 - beta1);
            auto vhat = v[i] / (1 - beta2);
            w[i] -= lr * rsqrtf(vhat + epsilon) * mhat;
        }
    }
}