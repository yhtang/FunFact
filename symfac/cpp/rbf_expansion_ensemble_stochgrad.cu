#include <type_traits>
#include "cuda_util.h"
#include "tea.h"
#include "tensor_view.h"

extern "C" {

__global__ void rbf_expansion_ensemble_stochgrad(
    float * __restrict ptr_A,
    float * __restrict ptr_u,
    float * __restrict ptr_v,
    float * __restrict ptr_a,
    float * __restrict ptr_b,
    float * __restrict ptr_L,
    float * __restrict ptr_du,
    float * __restrict ptr_dv,
    float * __restrict ptr_da,
    float * __restrict ptr_db,
    const uint rng_key,
    const int  n
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

    // volatile __shared__ float  a_cache[warp_per_block][r];

    // auto  a_warp = f_tensor_view( a_cache[i_warp_local], r);

    float  A_local;
    float dA_local;

    float db_local = 0;
    float loss_local = 0;

    __syncthreads();

    // #define __debug__ printf("thread %d line %d\n", threadIdx.x, __LINE__);
    #define __debug__

    for(int p = i_thread + i_block * thread_per_block; p < n; p += thread_per_block * block_per_inst) {

        auto ij = TEA::CBRNG<16>(rng_key, p);
        int i = ij.x % N;
        int j = ij.y % M;

        __debug__

        //----------------------------------------------------------------------
        // forward calculations
        // evaluate the RBF components
        __debug__

        A_local = b(i_instance);
        #pragma unroll (R)
        for(int k = 0; k < R; ++k) {
            auto d = u(i, k, i_instance) - v(j, k, i_instance);
            auto eij = expf(-d * d);
            A_local += a(k, i_instance) * eij;
        }

        __debug__

        //----------------------------------------------------------------------
        // loss calculations

        // constexpr static float loss_normalization = 1.f / (N * M);
        float loss_normalization = 1.f / n;
        auto delta = A_local - A(i, j);
        auto dA = 2 * loss_normalization * delta;
        dA_local = dA;
        db_local += dA;
        loss_local += loss_normalization * delta * delta;

        //----------------------------------------------------------------------
        // backward calculations

        __debug__

        for(int k = 0; k < R; ++k) {
            __debug__
            auto d = u(i, k, i_instance) - v(j, k, i_instance);
            auto eij = expf(-d * d);
            auto duv = -2.f * dA_local * a(k, i_instance) * eij * d;

            auto delta_u = duv;
            auto delta_v = -duv;
            atomicAdd(du.at(i, k, i_instance), delta_u);
            atomicAdd(dv.at(j, k, i_instance), delta_v);
            __debug__

            // auto delta_a = warp_sum(dA * eij);
            // if (lane == 0) {
            atomicAdd(da.at(k, i_instance), dA * eij);
            // }    
        }
    }

    __debug__

    // store results to global memory
    atomicAdd(db.at(i_instance), db_local);
    atomicAdd( L.at(i_instance), loss_local);
    __syncthreads();
    __debug__
}

}
