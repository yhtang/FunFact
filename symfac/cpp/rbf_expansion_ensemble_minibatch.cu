#include <type_traits>
#include "cuda_util.h"
#include "tea.h"
#include "tensor_view.h"

extern "C" {

__global__ void rbf_expansion_ensemble_minibatch(
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

    /*---------------------------------------------------
    a kernel call = an ensemble of instances
    an instance   = several CUDA blocks
    a block       = loop over tiles of the target matrix
    ---------------------------------------------------*/

    constexpr static int ilp              = ${ilp};
    constexpr static int thread_per_block = ${thread_per_block}; // == blockDim.x;
    constexpr static int block_per_inst   = ${block_per_inst};   // == gridDim.x;
    const int i_thread     = threadIdx.x;
    const int i_block      = blockIdx.x;
    const int i_instance   = blockIdx.y;
    const int lane         = i_thread % warp_size;

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

    volatile __shared__ float  a_cache[warp_per_block][r];

    auto  a_warp = f_tensor_view( a_cache[i_warp_local], r);

    float  A_local[ilp];
    float dA_local[ilp];

    float db_local = 0;
    float loss_local = 0;

    __syncthreads();

    // #define __debug__ printf("thread %d line %d\n", threadIdx.x, __LINE__);
    #define __debug__

    for(int tile_id = i_warp_inst; tile_id < tiles_n * tiles_m; tile_id += warp_per_inst) {

        // upper-left coordinate of the tile
        int I = tile_id % tiles_n * n;
        int J = tile_id / tiles_n * m;

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

            // prepare warp-level cache
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
                }
            }
            #pragma unroll
            par_for(p, m * r) {
                auto j = p % m;
                auto k = p / m;
                if (J + j < M && K + k < R) {
                    v_warp (j, k) = v(J + j, K + k, i_instance);
                }
            }
            #pragma unroll
            par_for(k, r) {
                a_warp (k) = a(K + k, i_instance);
            }
            __syncthreads();

            __debug__

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

                    auto delta_u = duv;
                    auto delta_v = -duv;
                    for(int mask = n; mask < warp_size; mask <<= 1) {
                        delta_u += __shfl_xor_sync(0xFFFFFFFF, delta_u, mask);
                    }
                    for(int mask = 1; mask < n; mask <<= 1) {
                        delta_v += __shfl_xor_sync(0xFFFFFFFF, delta_v, mask);
                    }
                    if (lane < n) {
                        atomicAdd(du.at(I + i, K + k, i_instance), delta_u);
                    }
                    if (lane % n == 0) {
                        atomicAdd(dv.at(J + j, K + k, i_instance), delta_v);
                    }

                    auto delta_a = warp_sum(dA * eij);
                    if (lane == 0) {
                        atomicAdd(da.at(K + k, i_instance), delta_a);
                    }
                    __debug__
                }
                __debug__
            }
            __syncthreads();

            __debug__
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
