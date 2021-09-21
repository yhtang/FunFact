#include <type_traits>
#include "cuda_util.h"
#include "tea.h"
#include "tensor_view.h"

#define SamplingNonZeros 0
#define SamplingZeros    1

extern "C" {

__global__ void rbf_expansion_ensemble_sparse_stochgrad_${GOAL}(
    uint  * __restrict rng_key,
    int   * __restrict hitmap,
    int2  * __restrict nz_indices,
    float * __restrict nz_values,
    float * __restrict ptr_u,
    float * __restrict ptr_v,
    float * __restrict ptr_a,
    float * __restrict ptr_b,
    float * __restrict ptr_L,
    float * __restrict ptr_du,
    float * __restrict ptr_dv,
    float * __restrict ptr_da,
    float * __restrict ptr_db,
    const int  n
) {
    constexpr static int E   = ${E};   // number of instances in ensemble
    constexpr static int N   = ${N};   // number of matrix rows
    constexpr static int M   = ${M};   // number of matrix columns
    constexpr static int R   = ${R};   // number of RBF components
    constexpr static int NNZ = ${NNZ}; // number of non-zeros in the target
    constexpr static int NH  = ${NH};  // number of entries in the hitmap

    /*---------------------------------------------------
    a kernel call = an ensemble of instances
    an instance   = several CUDA blocks
    a block       = loop over tiles of the target matrix
    ---------------------------------------------------*/

    constexpr static int thread_per_block = ${thread_per_block}; // == blockDim.x;
    constexpr static int block_per_inst   = ${block_per_inst};   // == gridDim.x;
    const int i_thread   = threadIdx.x;
    const int i_block    = blockIdx.x;
    const int i_instance = blockIdx.y;
    const int lane       = i_thread % warp_size;

    auto  u = tensor_view<MemoryLayout::Fortran>(ptr_u,  N,    R, E);
    auto du = tensor_view<MemoryLayout::Fortran>(ptr_du, N,    R, E);
    auto  v = tensor_view<MemoryLayout::Fortran>(ptr_v,     M, R, E);
    auto dv = tensor_view<MemoryLayout::Fortran>(ptr_dv,    M, R, E);
    auto  a = tensor_view<MemoryLayout::Fortran>(ptr_a,        R, E);
    auto da = tensor_view<MemoryLayout::Fortran>(ptr_da,       R, E);
    auto  b = tensor_view<MemoryLayout::Fortran>(ptr_b,           E);
    auto db = tensor_view<MemoryLayout::Fortran>(ptr_db,          E);
    auto  L = tensor_view<MemoryLayout::Fortran>(ptr_L,           E);

    const uint key = rng_key[i_instance];

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

        __debug__

        #if ${GOAL} == SamplingNonZeros

            auto q   = TEA::get_1x32<8>(key, p) % NNZ;
            auto Aij = nz_values [q];
            auto ij  = nz_indices[q];
            int  i   = ij.x;
            int  j   = ij.y;

            auto set_bit = [](auto H, auto i, auto j){
                auto h = TEA::get_1x32<8>(i, j) % NH;
                atomicOr(H + h / 32, 1 << (h % 32));
            };
            set_bit(hitmap, i, j);

        #elif ${GOAL} == SamplingZeros

            int i, j;
            auto Aij = 0.f;

            auto is_bit_set = [](auto H, auto i, auto j){
                auto h = TEA::get_1x32<8>(i, j) % NH;
                return H[h / 32] & (1 << (h % 32));
            };
            auto tea = make_uint2(key, p);
            do {
                tea = TEA::get_2x32<8>(tea.x, tea.y);
                i = tea.x % N;
                j = tea.y % M;
                __debug__
            } while (is_bit_set(hitmap, i, j));

        #endif

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
        auto delta = A_local - Aij;
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

            // atomicAdd(da.at(k, i_instance), dA * eij);
            auto delta_a = warp_sum(dA * eij);
            if (lane == 0) {
                atomicAdd(da.at(k, i_instance), delta_a);
            }
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
