/**********************************************************************************

 Copyright (c) 2026-2026 Linus Reuter
 Copyright (c) 2023-2025 Patrick Steil
 Copyright (c) 2019-2022 KIT ITI Algorithmics Group

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
#include <emmintrin.h>
#endif

#include <cassert>
#include <cstring>
#include <vector>

#include "../../../DataStructures/TripBased/Data.h"
#include "../../../ExternalLibs/aligned_allocator.h"

namespace TripBased {

// ---------------------------------------------------------------------------
// Platform-agnostic 128-bit vector helpers
// ---------------------------------------------------------------------------
#if defined(__aarch64__) || defined(__arm__)
// ARM NEON path ----------------------------------------------------------
using simd128_t = uint8x16_t;

inline simd128_t simd_set1_u8(uint8_t val) { return vdupq_n_u8(val); }
inline simd128_t simd_max_u8(simd128_t a, simd128_t b) { return vmaxq_u8(a, b); }
inline simd128_t simd_min_u8(simd128_t a, simd128_t b) { return vminq_u8(a, b); }

// Build a mask where bytes 0..round-2 are 0xFF and the rest are 0x00.
inline simd128_t makeMask(uint8_t round) {
  uint8_t buf[16] = {};
  for (int i = 0; i < static_cast<int>(round) - 1 && i < 16; ++i)
    buf[i] = 0xFF;
  return vld1q_u8(buf);
}

#else
// x86 SSE2 path ----------------------------------------------------------
using simd128_t = __m128i;

inline simd128_t simd_set1_u8(uint8_t val) { return _mm_set1_epi8(static_cast<char>(val)); }
inline simd128_t simd_max_u8(simd128_t a, simd128_t b) { return _mm_max_epu8(a, b); }
inline simd128_t simd_min_u8(simd128_t a, simd128_t b) { return _mm_min_epu8(a, b); }

static constexpr const __m128i MAX_MASKS[16] = {0x0000000000000000,
                                                0x0000000000000000,
                                                0x00000000000000FF,
                                                0x0000000000000000,
                                                0x000000000000FFFF,
                                                0x0000000000000000,
                                                0x0000000000FFFFFF,
                                                0x0000000000000000,
                                                0x00000000FFFFFFFF,
                                                0x0000000000000000,
                                                0x000000FFFFFFFFFF,
                                                0x0000000000000000,
                                                0x0000FFFFFFFFFFFF,
                                                0x0000000000000000,
                                                0x00FFFFFFFFFFFFFF,
                                                0x0000000000000000,
                                                -1,
                                                0x0000000000000000,
                                                -1,
                                                0x00000000000000FF,
                                                -1,
                                                0x000000000000FFFF,
                                                -1,
                                                0x0000000000FFFFFF,
                                                -1,
                                                0x00000000FFFFFFFF,
                                                -1,
                                                0x000000FFFFFFFFFF,
                                                -1,
                                                0x0000FFFFFFFFFFFF,
                                                -1,
                                                0x00FFFFFFFFFFFFFF};

inline simd128_t makeMask(uint8_t round) { return MAX_MASKS[round - 1]; }

#endif  // platform select

// ---------------------------------------------------------------------------
// ProfileReachedIndexSIMD
// ---------------------------------------------------------------------------

//! Allows to check whether we already reached a certain point in a route / trip
//! / position given a number of rounds. Lookup is fast, but updating is slow.
//! This ReachedIndex is used for the TB::ProfileQuery. It uses SIMD intrisics
//! to allow for fast updates.
class ProfileReachedIndexSIMD {
 private:
  //! This union holds the values (aligned to use SIMD intrisics)
  union alignas(16) ReachedElement {
    ReachedElement() {}
    simd128_t mValues;
    u_int8_t values[16];
  };

 public:
  ProfileReachedIndexSIMD(const Data& data)
      : data(data),
        defaultLabels(data.numberOfTrips()),
        labels(data.numberOfTrips()) {
    for (TripId trip(0); trip < data.numberOfTrips(); ++trip) {
      std::fill(std::begin(defaultLabels[trip].values),
                std::end(defaultLabels[trip].values),
                data.numberOfStopsInTrip(trip));
    }
  };

  inline void clear() noexcept { labels = defaultLabels; }

  inline bool alreadyReached(const TripId trip, const u_int8_t position,
                             const uint8_t round = 1) noexcept {
    assert(data.isTrip(trip));
    assert(0 < round);
    assert(round < 16);

    return getPosition(trip, round) <= position;
  }

  inline void update(const TripId trip, const u_int8_t position,
                     const uint8_t round = 1) noexcept {
    assert(data.isTrip(trip));
    assert(0 < round);
    assert(round < 16);

    simd128_t mask = makeMask(round);

    const simd128_t FILTER = simd_max_u8(simd_set1_u8(position), mask);

    // Iterate over all trips either until the last trip OR if we already have a
    // trip with a position at least as good
    for (TripId tr(trip);
         tr < data.firstTripOfRoute[data.routeOfTrip[trip] + 1] &&
         getPosition(tr, round) > position;
         ++tr)
      labels[tr].mValues = simd_min_u8(labels[tr].mValues, FILTER);
  }

  inline u_int8_t& operator()(const TripId trip,
                              const uint8_t round = 1) noexcept {
    return getPosition(trip, round);
  }

 private:
  inline u_int8_t& getPosition(const TripId trip,
                               const uint8_t round = 1) noexcept {
    return labels[trip].values[round - 1];
  }

  //! Returns the filter mask to use
  inline simd128_t getMask(const uint8_t round = 1) const noexcept {
    assert(0 < round);
    assert(round < 16);

    return makeMask(round);
  }

  const Data& data;

  std::vector<ReachedElement,
              aligned_allocator<ReachedElement, alignof(ReachedElement)>>
      defaultLabels;
  std::vector<ReachedElement,
              aligned_allocator<ReachedElement, alignof(ReachedElement)>>
      labels;
};

}  // namespace TripBased
