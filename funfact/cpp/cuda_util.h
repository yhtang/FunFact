constexpr static int warp_size = 32;

template<class T> __device__ T warp_sum(T value) {
    #pragma unroll
    for(int p = (warp_size >> 1); p >= 1 ; p >>= 1) value += __shfl_xor_sync(0xFFFFFFFF, value, p);
    return value;
}
