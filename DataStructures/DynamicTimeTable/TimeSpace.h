#pragma once

#include "../../Helpers/Types.h"

namespace DynamicTimeTable {

/**
 * The two time spaces of the dynamic pipeline, and the only place either is converted.
 *
 *  - REAL time (`a`, `d`): wall clock, as a GTFS/GTFS-RT feed reports it. This is what
 *    `PendingUpdates` carries and what `DynamicTimeTable::Data` stores. Every invariant --
 *    `a <= d` per event, arrivals and departures nondecreasing along a trip -- holds in this
 *    space and only in this space, so all enforcement reads stored times directly.
 *
 *  - BOARD DEADLINE (`d_hat = d - c(stop)`, `c = minTransferTime`): the latest arrival time
 *    from which this departure is still catchable. Not a time -- a threshold on someone
 *    else's arrival. Every legitimate use has the shape `foreignArrival + footpath <= d_hat`;
 *    comparing it against anything on its own trip is a bug, because `d_hat` is not monotone
 *    along a trip. It exists only in the exported `DynamicQueryData` (`eventDepTimes` and
 *    `routeLabels[r].departureTimes`), which is where `TransferDiscovery`, `isUTurn` and the
 *    minimization read it -- matching the convention of the static TripBased builder, so a
 *    dynamically built QueryData stays comparable with a statically built one.
 *
 * Arrivals are identical in both spaces; the implicit *arrival* buffer encoding is not used.
 *
 * `toBoardDeadline` is applied exactly once, on the export write in `BuildQueryData`, and
 * undone exactly once, in that file's validator. Nothing else may convert.
 */
namespace TimeSpace {

// Real departure -> board deadline. `noTime` (boarding forbidden) is not a time and passes through.
inline Time toBoardDeadline(const Time realDeparture, const int minTransferTime) noexcept {
    if (realDeparture == noTime) return noTime;
    // Departures never precede the change time in practice; clamp rather than wrap the unsigned.
    const auto d = static_cast<std::int64_t>(realDeparture);
    const auto c = static_cast<std::int64_t>(minTransferTime);
    return Time(static_cast<std::uint32_t>(d > c ? d - c : 0));
}

// Board deadline -> real departure. Inverse of the above on every value it did not clamp.
inline Time toRealDeparture(const Time boardDeadline, const int minTransferTime) noexcept {
    if (boardDeadline == noTime) return noTime;
    return Time(static_cast<std::uint32_t>(static_cast<std::int64_t>(boardDeadline) +
                                           static_cast<std::int64_t>(minTransferTime)));
}

}  // namespace TimeSpace
}  // namespace DynamicTimeTable
