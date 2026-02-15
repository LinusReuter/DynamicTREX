/**********************************************************************************

 Copyright (c) 2023-2025 Patrick Steil

 MIT License

 Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**********************************************************************************/
#pragma once

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

#include <cstdint>
#include <iomanip>
#include <iostream>

// ---------------------------------------------------------------------------
// ARM NEON path: emulate 256-bit with two 128-bit NEON registers
// ---------------------------------------------------------------------------
#if defined(__aarch64__) || defined(__arm__)

union Holder {
  struct { uint16x8_t lo; uint16x8_t hi; };
  std::uint16_t arr[16];
};

struct SIMD16u {
  Holder v;

  SIMD16u() noexcept = default;
  SIMD16u(uint16x8_t lo, uint16x8_t hi) noexcept { v.lo = lo; v.hi = hi; }
  SIMD16u(uint16_t scalar) noexcept { v.lo = vdupq_n_u16(scalar); v.hi = vdupq_n_u16(scalar); }
  void fill(uint16_t scalar) noexcept { v.lo = vdupq_n_u16(scalar); v.hi = vdupq_n_u16(scalar); }

  static SIMD16u load(const uint16_t *ptr) noexcept {
    SIMD16u r;
    r.v.lo = vld1q_u16(ptr);
    r.v.hi = vld1q_u16(ptr + 8);
    return r;
  }
  void store(uint16_t *ptr) const noexcept {
    vst1q_u16(ptr, v.lo);
    vst1q_u16(ptr + 8, v.hi);
  }

  uint16_t &operator[](std::size_t i) noexcept { return v.arr[i & 15]; }
  const uint16_t &operator[](std::size_t i) const noexcept { return v.arr[i & 15]; }

  SIMD16u operator+(const SIMD16u &o) const noexcept {
    return SIMD16u(vaddq_u16(v.lo, o.v.lo), vaddq_u16(v.hi, o.v.hi));
  }
  SIMD16u operator-(const SIMD16u &o) const noexcept {
    return SIMD16u(vsubq_u16(v.lo, o.v.lo), vsubq_u16(v.hi, o.v.hi));
  }

  SIMD16u operator&(const SIMD16u &o) const noexcept {
    return SIMD16u(vandq_u16(v.lo, o.v.lo), vandq_u16(v.hi, o.v.hi));
  }
  SIMD16u operator|(const SIMD16u &o) const noexcept {
    return SIMD16u(vorrq_u16(v.lo, o.v.lo), vorrq_u16(v.hi, o.v.hi));
  }
  SIMD16u operator^(const SIMD16u &o) const noexcept {
    return SIMD16u(veorq_u16(v.lo, o.v.lo), veorq_u16(v.hi, o.v.hi));
  }

  SIMD16u sll(int bits) const noexcept {
    int16x8_t shift = vdupq_n_s16(static_cast<int16_t>(bits));
    return SIMD16u(vshlq_u16(v.lo, shift), vshlq_u16(v.hi, shift));
  }
  SIMD16u srl(int bits) const noexcept {
    int16x8_t shift = vdupq_n_s16(static_cast<int16_t>(-bits));
    return SIMD16u(vshlq_u16(v.lo, shift), vshlq_u16(v.hi, shift));
  }

  SIMD16u cmpeq(const SIMD16u &o) const noexcept {
    return SIMD16u(vceqq_u16(v.lo, o.v.lo), vceqq_u16(v.hi, o.v.hi));
  }

  // max: sets v to max(v, o), returns mask where v was already >= o (all-ones per lane)
  SIMD16u maxmask(const SIMD16u &o) noexcept {
    uint16x8_t mlo = vmaxq_u16(v.lo, o.v.lo);
    uint16x8_t mhi = vmaxq_u16(v.hi, o.v.hi);
    uint16x8_t eq0lo = vceqq_u16(mlo, v.lo);
    uint16x8_t eq0hi = vceqq_u16(mhi, v.hi);
    v.lo = mlo;
    v.hi = mhi;
    return SIMD16u(eq0lo, eq0hi);
  }

  // min: sets v to min(v, o), returns mask where v was already <= o (all-ones per lane)
  SIMD16u minmask(const SIMD16u &o) noexcept {
    uint16x8_t mlo = vminq_u16(v.lo, o.v.lo);
    uint16x8_t mhi = vminq_u16(v.hi, o.v.hi);
    uint16x8_t eq0lo = vceqq_u16(mlo, v.lo);
    uint16x8_t eq0hi = vceqq_u16(mhi, v.hi);
    v.lo = mlo;
    v.hi = mhi;
    return SIMD16u(eq0lo, eq0hi);
  }

  void blend(const SIMD16u &other, const SIMD16u &mask) noexcept {
    // vbslq: for each bit, selects from first arg where mask is 1, second where 0
    v.lo = vbslq_u16(mask.v.lo, v.lo, other.v.lo);
    v.hi = vbslq_u16(mask.v.hi, v.hi, other.v.hi);
  }
};

// ---------------------------------------------------------------------------
// x86 AVX2 path: original implementation
// ---------------------------------------------------------------------------
#else

union Holder {
  __m256i reg;
  std::uint16_t arr[16];
};

struct SIMD16u {
  Holder v;

  SIMD16u() noexcept = default;
  explicit SIMD16u(__m256i x) noexcept { v.reg = x; }
  SIMD16u(uint16_t scalar) noexcept { v.reg = _mm256_set1_epi16(scalar); }
  void fill(uint16_t scalar) noexcept { v.reg = _mm256_set1_epi16(scalar); }

  static SIMD16u load(const uint16_t *ptr) noexcept {
    return SIMD16u(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr)));
  }
  void store(uint16_t *ptr) const noexcept {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), v.reg);
  }

  uint16_t &operator[](std::size_t i) noexcept { return v.arr[i & 15]; }
  const uint16_t &operator[](std::size_t i) const noexcept {
    return v.arr[i & 15];
  }

  SIMD16u operator+(const SIMD16u &o) const noexcept {
    return SIMD16u(_mm256_add_epi16(v.reg, o.v.reg));
  }
  SIMD16u operator-(const SIMD16u &o) const noexcept {
    return SIMD16u(_mm256_sub_epi16(v.reg, o.v.reg));
  }

  SIMD16u operator&(const SIMD16u &o) const noexcept {
    return SIMD16u(_mm256_and_si256(v.reg, o.v.reg));
  }
  SIMD16u operator|(const SIMD16u &o) const noexcept {
    return SIMD16u(_mm256_or_si256(v.reg, o.v.reg));
  }
  SIMD16u operator^(const SIMD16u &o) const noexcept {
    return SIMD16u(_mm256_xor_si256(v.reg, o.v.reg));
  }

  SIMD16u sll(int bits) const noexcept {
    return SIMD16u(_mm256_slli_epi16(v.reg, bits));
  }
  SIMD16u srl(int bits) const noexcept {
    return SIMD16u(_mm256_srli_epi16(v.reg, bits));
  }

  SIMD16u cmpeq(const SIMD16u &o) const noexcept {
    return SIMD16u(_mm256_cmpeq_epi16(v.reg, o.v.reg));
  }

  __m256i max(const SIMD16u &o) noexcept {
    __m256i m = _mm256_max_epu16(v.reg, o.v.reg);
    __m256i eq0 = _mm256_cmpeq_epi16(m, v.reg);
    v.reg = m;
    return eq0;
  }

  __m256i min(const SIMD16u &o) noexcept {
    __m256i m = _mm256_min_epu16(v.reg, o.v.reg);
    __m256i eq0 = _mm256_cmpeq_epi16(m, v.reg);
    v.reg = m;
    return eq0;
  }

  void blend(const SIMD16u &other, __m256i mask) noexcept {
    v.reg = _mm256_blendv_epi8(other.v.reg, v.reg, mask);
  }
};

#endif  // platform select

inline void printSIMD(const char *name, const SIMD16u &x) {
  std::cout << std::setw(10) << name << ": [";
  for (int i = 0; i < 16; ++i) {
    std::cout << x.v.arr[i] << (i < 15 ? ", " : "");
  }
  std::cout << "]\n";
}