#include <cstdint>

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
