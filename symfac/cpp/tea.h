struct TEA {
    constexpr static uint _K0 = 0xA341316C;
    constexpr static uint _K1 = 0xC8013EA4;
    constexpr static uint _K2 = 0xAD90777D;
    constexpr static uint _K3 = 0x7E95761E;
    constexpr static uint _DT = 0x9E3779B9;
    
    template<int N> __inline__ __host__ __device__
    static void _pass(uint &_0, uint &_1, uint sum = 0)
    {
        sum += _DT;
        _0 += ( ( _1 << 4 ) + _K0 ) ^ ( _1 + sum ) ^ ( ( _1 >> 5 ) + _K1 );
        _1 += ( ( _0 << 4 ) + _K2 ) ^ ( _0 + sum ) ^ ( ( _0 >> 5 ) + _K3 );
        _pass<N - 1>(_0, _1, sum);
    }

    template<> __inline__ __host__ __device__
    static void _pass<0>(uint &_0, uint &_1, uint sum) {}

    template<int N> __inline__ __host__ __device__
    static uint2 TEA(uint key, uint counter)
    {
        _pass<N>(key, counter);
        return make_uint2(key, counter);
    }
};
