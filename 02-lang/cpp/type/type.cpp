#include <iostream>

using namespace std;

typename <int Bytes> struct wide_container;
template <> struct wide_container<1> {using type = uint8_t;};
template <> struct wide_container<2> {using type = uint16_t;};
template <> struct wide_container<4> {using type = uint32_t;};
template <> struct wide_container<8> {using type = uint64_t;};
template <> struct wide_container<16> {using type = uint4;};
template<int Bytes>using wide_containter_t = typename wide_
