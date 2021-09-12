#include <cstdint>
#include <type_traits>

constexpr static int warp_size = 32;

template<class T> __device__ T warp_sum(T value) {
    #pragma unroll
    for(int p = (warp_size >> 1); p >= 1 ; p >>= 1) value += __shfl_xor_sync(0xFFFFFFFF, value, p);
    return value;
}

enum class MemoryLayout {
    C=0,
    Fortran=1
};

template<MemoryLayout memory_layout, class type, int ndim>
struct tensor_view_t {
    using size_type = std::int32_t;
    using element_type = std::remove_const_t<type>;
    using pointer_type = std::add_pointer_t<element_type>;

    pointer_type const _ptr;
    size_type const _shape[ndim];

    template<class ...I>
    __host__ __device__ __inline__
    tensor_view_t(pointer_type ptr, I ... shape):
        _ptr(ptr), _shape {shape...} {}

    template<MemoryLayout layout, std::enable_if_t<layout == MemoryLayout::C, bool> = true>
    __host__ __device__ __inline__
    size_type linearize(size_type const idx[], size_type const shape[]) {
        size_type offset = 0;
        for(int i = 0; i < ndim; ++i) {
            offset = offset * _shape[i] + idx[i];
        }
        return offset;
    }

    template<MemoryLayout layout, std::enable_if_t<layout == MemoryLayout::Fortran, bool> = true>
    __host__ __device__ __inline__
    size_type linearize(size_type const idx[], size_type const shape[]) {
        size_type offset = 0;
        for(int i = ndim - 1; i >= 0; --i) {
            offset = offset * _shape[i] + idx[i];
        }
        return offset;
    }

    template<class ...I>
    __host__ __device__ __inline__
    pointer_type at(I ... indices) {
        static_assert(sizeof...(indices) == ndim);
        // multi-index address is essentially polynomial evaluation
        size_type idx[ndim] {indices...};
        return _ptr + linearize<memory_layout>(idx, _shape);
    }

    template<class ...I>
    __host__ __device__ __inline__
    element_type & operator () (I ... indices) {
        return *at(indices...);
    }
};

template<MemoryLayout memory_layout, class T, class ... I>
__host__ __device__ __inline__
auto tensor_view(T * ptr, I ... shape) {
    return tensor_view_t<memory_layout, T, sizeof...(I)>(ptr, shape...);
}

template<class T, class ... I>
__host__ __device__ __inline__
auto c_tensor_view(T * ptr, I ... shape) {
    return tensor_view_t<MemoryLayout::C, T, sizeof...(I)>(ptr, shape...);
}

template<class T, class ... I>
__host__ __device__ __inline__
auto f_tensor_view(T * ptr, I ... shape) {
    return tensor_view_t<MemoryLayout::Fortran, T, sizeof...(I)>(ptr, shape...);
}

extern "C" {

__global__ void rbf_expansion_ensemble(
    float * __restrict ptr_A,
    float * __restrict ptr_u,
    float * __restrict ptr_v,
    float * __restrict ptr_a,
    float * __restrict ptr_b,
    float * __restrict ptr_L,
    float * __restrict ptr_du,
    float * __restrict ptr_dv,
    float * __restrict ptr_da,
    float * __restrict ptr_db
) {
    constexpr static int E = ${E};  // number of instances in ensemble
    constexpr static int N = ${N};  // number of matrix rows
    constexpr static int M = ${M};  // number of matrix columns
    constexpr static int R = ${R};  // number of RBF components
    constexpr static int n = ${n};
    constexpr static int m = ${m};
    constexpr static int r = ${r};
    constexpr static int tiles_n = (N + n - 1) / n;
    constexpr static int tiles_m = (M + m - 1) / m;

    static_assert(
        n * m > 0 && n * m % 32 == 0,
        "Tile size in number of elements must be non-negative multiple of 32."
    );

    /*---------------------------------------------------
    a kernel call = an ensemble of instances
    an instance   = several CUDA blocks
    a block       = loop over tiles of the target matrix
    ---------------------------------------------------*/

    constexpr static int thread_per_block = ${thread_per_block}; // == blockDim.x;
    constexpr static int block_per_inst   = ${block_per_inst};   // == gridDim.x;
    constexpr static int warp_per_block   = thread_per_block / warp_size;
    constexpr static int warp_per_inst    = warp_per_block * block_per_inst;
    constexpr static int elem_per_thread  = n * m / warp_size;
    const int i_thread     = threadIdx.x;
    const int i_block      = blockIdx.x;
    const int i_instance   = blockIdx.y;
    const int lane         = i_thread % warp_size;
    const int i_warp_local = i_thread / warp_size;
    const int i_warp_inst  = i_warp_local + i_block * warp_per_block;

    // const int warp_id_local = threadIdx.x / warp_size;
    // const int warp_id_global = warp_id_local + blockIdx.x * blockDim.x / warp_size;

    auto  A = tensor_view<MemoryLayout::Fortran>(ptr_A,  N, M      );
    auto  u = tensor_view<MemoryLayout::Fortran>(ptr_u,  N,    R, E);
    auto du = tensor_view<MemoryLayout::Fortran>(ptr_du, N,    R, E);
    auto  v = tensor_view<MemoryLayout::Fortran>(ptr_v,     M, R, E);
    auto dv = tensor_view<MemoryLayout::Fortran>(ptr_dv,    M, R, E);
    auto  a = tensor_view<MemoryLayout::Fortran>(ptr_a,        R, E);
    auto da = tensor_view<MemoryLayout::Fortran>(ptr_da,       R, E);
    auto  b = tensor_view<MemoryLayout::Fortran>(ptr_b,           E);
    auto db = tensor_view<MemoryLayout::Fortran>(ptr_db,          E);
    auto  L = tensor_view<MemoryLayout::Fortran>(ptr_L,           E);

    // volatile __shared__ float  A_cache[warp_per_block][n * m    ];
    volatile __shared__ float  u_cache[warp_per_block][n     * r];
    volatile __shared__ float du_cache[warp_per_block][n     * r];
    volatile __shared__ float  v_cache[warp_per_block][    m * r];
    volatile __shared__ float dv_cache[warp_per_block][    m * r];
    volatile __shared__ float  a_cache[warp_per_block][        r];
    volatile __shared__ float da_cache[warp_per_block][        r];

    // auto  A_warp = f_tensor_view( A_cache[i_warp_local], n, m);
    // auto dA_warp = f_tensor_view( A_cache[i_warp_local], n, m);  // reuse the buffer of A
    auto  u_warp = f_tensor_view( u_cache[i_warp_local], n, r);
    // auto du_warp = f_tensor_view(du_cache[i_warp_local], n, r);
    auto  v_warp = f_tensor_view( v_cache[i_warp_local], m, r);
    // auto dv_warp = f_tensor_view(dv_cache[i_warp_local], m, r);
    auto  a_warp = f_tensor_view( a_cache[i_warp_local], r);
    // auto da_warp = f_tensor_view(da_cache[i_warp_local], r);

    float  A_local[elem_per_thread];
    float dA_local[elem_per_thread];

    float db_local = 0;
    float loss_local = 0;

    __syncthreads();

    // #define __debug__ printf("thread %d line %d\n", threadIdx.x, __LINE__);
    #define __debug__

    for(int tile_id = i_warp_inst; tile_id < tiles_n * tiles_m; tile_id += warp_per_inst) {

        // upper-left coordinate of the tile
        int I = tile_id % tiles_n * n;
        int J = tile_id / tiles_n * m;

        // if (lane == 0) {
        //     printf("IJ: %d %d, tiles_n, tiles_m %d %d,\n", I, J, tiles_n, tiles_m);
        // }

        __debug__

        #define par_for(iter, range) \
            for(int iter = lane; iter < range; iter += warp_size)
        #define par_enumerate(idx, iter, range) \
            for(int iter = lane, idx = 0; iter < range; iter += warp_size, ++idx)

        //----------------------------------------------------------------------
        // forward calculations

        #pragma unroll
        par_enumerate(q, p, n * m) {
            A_local[q] = b(i_instance);
        }

        __debug__

        for(int K = 0; K < R; K += r) { // loop over RBF components

            // prepare block-level cache
            #pragma unroll
            par_for(p, n * r) {
                auto i = p % n;
                auto k = p / n;
                if (I + i < N && K + k < R) {
                    u_warp(i, k) = u(I + i, K + k, i_instance);
                }
            }
            #pragma unroll
            par_for(p, m * r) {
                auto j = p % m;
                auto k = p / m;
                if (J + j < M && K + k < R) {
                    v_warp(j, k) = v(J + j, K + k, i_instance);
                }
            }
            #pragma unroll
            par_for(k, r) {
                if (K + k < R) {
                    a_warp(k) = a(K + k, i_instance);
                }
            }
            __syncthreads();

            __debug__

            // evaluate the RBF components
            #pragma unroll (elem_per_thread)
            par_enumerate(q, p, n * m) {
                auto i = p % n;
                auto j = p / n;
                if (I + i < N && J + j < M) {
                    for(int k = 0; k < min(r, R - K); ++k) {
                        auto d = u_warp(i, k) - v_warp(j, k);
                        auto eij = expf(-d * d);
                        A_local[q] += a_warp(k) * eij;
                    }
                }
            }
            __syncthreads();
        }

        __debug__

        //----------------------------------------------------------------------
        // loss calculations

        // TODO: reduction for total loss
        #pragma unroll (elem_per_thread)
        par_enumerate(q, p, n * m) {
            auto i = p % n;
            auto j = p / n;
            if (I + i < N && J + j < M) {
                // TODO: templaterize
                constexpr static float loss_normalization = 1.f / (N * M);
                auto delta = A_local[q] - A(I + i, J + j);
                auto dA = 2 * loss_normalization * delta;
                dA_local[q] = dA;
                db_local   += dA;
                loss_local += loss_normalization * delta * delta;
            }
        }
        __syncthreads();

        //----------------------------------------------------------------------
        // backward calculations

        __debug__

        for(int K = 0; K < R; K += r) { // loop over RBF components

            // prepare block-level cache
            #pragma unroll
            par_for(p, n * r) {
                auto i = p % n;
                auto k = p / n;
                if (I + i < N && K + k < R) {
                    u_warp (i, k) = u(I + i, K + k, i_instance);
                    // du_warp(i, k) = 0;
                }
            }
            #pragma unroll
            par_for(p, m * r) {
                auto j = p % m;
                auto k = p / m;
                if (J + j < M && K + k < R) {
                    v_warp (j, k) = v(J + j, K + k, i_instance);
                    // dv_warp(j, k) = 0;
                }
            }
            #pragma unroll
            par_for(k, r) {
                a_warp (k) = a(K + k, i_instance);
                // da_warp(k) = 0;
            }
            __syncthreads();

            __debug__

            // constexpr static int cols_per_wave = warp_size / n;

            #pragma unroll
            par_enumerate(q, p, n * m) {
                auto i = p % n;
                auto j = p / n;
                bool active = I + i < N && J + j < M;
                auto dA = active ? dA_local[q] : 0.f;
                __debug__
                for(int k = 0; k < min(r, R - K); ++k) {
                    __debug__
                    auto d = u_warp(i, k) - v_warp(j, k);
                    auto eij = active ? expf(-d * d) : 0.f;
                    auto duv = active ? (-2.f * dA * a_warp(k) * eij * d) : 0.f;
                    // printf("i %d j %d n %d m %d p %d active %d duv %f\n", i, j, n, m, p, active, duv);

                    auto delta_u = duv;
                    auto delta_v = -duv;
                    for(int mask = n; mask < warp_size; mask <<= 1) {
                        delta_u += __shfl_xor_sync(0xFFFFFFFF, delta_u, mask);
                    }
                    for(int mask = 1; mask < n; mask <<= 1) {
                        delta_v += __shfl_xor_sync(0xFFFFFFFF, delta_v, mask);
                    }
                    if (lane < n) {
                        // du_warp(i, k) += delta_u;
                        atomicAdd(du.at(I + i, K + k, i_instance), delta_u);
                    }
                    if (lane % n == 0) {
                        // dv_warp(j, k) += delta_v;
                        atomicAdd(dv.at(J + j, K + k, i_instance), delta_v);
                    }

                    // for(int m = 1; m < n; m <<= 1) {
                    //     delta_u += __shfl_xor_sync(m, delta_u, 0xFFFFFFFF);
                    // }
                    // if (lane % n)

                    // if (active) atomicAdd(du_warp.at(i, k),  duv);
                    // printf("instance %d block %d thread %d write to du %d, %d\n", i_instance, i_block, i_thread, i, k);
                    // __syncthreads();
                    // if (active) atomicAdd(dv_warp.at(j, k), -duv);
                    // // printf("instance %d block %d thread %d write to dv %d, %d\n", i_instance, i_block, i_thread, j, k);
                    // __syncthreads();
                    auto delta_a = warp_sum(dA * eij);
                    // if (lane == 0) atomicAdd((float*)da_warp.at(k), da);
                    // if (lane == 0) da_warp(k) += da;
                    if (lane == 0) {
                        atomicAdd(da.at(K + k, i_instance), delta_a);
                    }
                    __debug__
                }
                __debug__
            }
            __syncthreads();

            __debug__

            // store tile/depth results to global memory
            #if 0
            #pragma unroll
            par_for(p, n * r) {
                auto i = p % n;
                auto k = p / n;
                if (I + i < N && K + k < R) {
                    // printf("thread %d adding %f @ (%d, %d) to du[%d, %d, %d]\n", threadIdx.x, du_warp(i, k), i, k, I + i, K + k, i_instance);
                    atomicAdd(du.at(I + i, K + k, i_instance), du_warp(i, k));
                }
            }
            #pragma unroll
            par_for(p, m * r) {
                auto j = p % m;
                auto k = p / m;
                if (J + j < M && K + k < R) {
                    atomicAdd(dv.at(J + j, K + k, i_instance), dv_warp(j, k));
                }
            }
            #pragma unroll
            par_for(k, r) {
                atomicAdd(da.at(K + k, i_instance), da_warp(k));
            }
            #endif
        }
        __debug__

    }

    __debug__

    // store results to global memory
    atomicAdd(db.at(i_instance), db_local);
    atomicAdd( L.at(i_instance), loss_local);
    __syncthreads();
    __debug__

}

}