#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "../../../Helpers/Types.h"

namespace DynamicTB::Preprocessing {

// Work-quantity counters are compile-time gated to keep the timed hot path
// uncontaminated. Build the "detail" measurement binary with
// -DDYN_COLLECT_TRANSFER_STATS to populate counters.
#ifdef DYN_COLLECT_TRANSFER_STATS
inline constexpr bool collectTransferStats = true;
#else
inline constexpr bool collectTransferStats = false;
#endif

/**
 * @brief Edge metadata stored in the transfer store.
 * Keep this minimal; transfer validity is derived from the timetable.
 *
 * `rank` is the TREX r(t) annotation: the highest partition level+1 for which this
 * transfer is still needed. It is only meaningful while `isMinimized` is true (the
 * reduced set is what the customization ranks) and is reset to 0 whenever an edge
 * leaves the reduced set. Over-approximating a rank is safe (the query keeps an edge
 * it could have pruned); under-approximating is not.
 *
 * Both fields fit in the padding `OutEdge` already had, so this costs no memory --
 * see the static_assert in ITransferStore-instantiating code.
 */
struct TransferMeta {
    bool isMinimized{false};  // true if kept by minimization
    std::uint8_t rank{0};     // TREX r(t); 0 = local only
};

/**
 * @brief Thread-local label structure matching StopEventGraphBuilder logic.
 */
struct StopLabel {
    int arrivalTime{std::numeric_limits<int>::max()};
    int timestamp{0};

    inline void checkTimestamp(const int newTimestamp) noexcept {
        if (timestamp != newTimestamp) {
            arrivalTime = std::numeric_limits<int>::max();
            timestamp = newTimestamp;
        }
    }

    inline void update(const int newTimestamp, const int newArrivalTime) noexcept {
        checkTimestamp(newTimestamp);
        arrivalTime = std::min(arrivalTime, newArrivalTime);
    }
};

/**
 * @brief A trip flagged for re-minimization, paired with the highest stop index at which its
 * keep-decisions can differ from the previous run (the "warm-start" boundary). Stops above
 * this index keep their previous flags; the minimizer only replays their kept edges to
 * rebuild StopLabels. See TransferMinimizer::reduceTransfersForTrip().
 */
using MinTarget = std::pair<TripId, StopIndex>;

/**
 * @brief A domination cleanup deferred out of the parallel incoming phase.
 */
struct PendingDominationCleanup {
    PersistentStopEventId fromEvent;
    StopEventId flatToEvent;
};

/**
 * @brief A rank raise produced by the TREX customization, in persistent (store) space.
 * Written back into TransferMeta::rank; deliberately sparse, so a customization that
 * touches few cells does not pay an O(reduced edges) write-back.
 */
struct RankRaise {
    PersistentStopEventId from;
    PersistentStopEventId to;
    std::uint8_t rank;
};

// Sentinel nowSeconds that disables the (compile-gated) time cutoff: no arrival is < INT_MIN.
inline constexpr int noTimeCutoff = std::numeric_limits<int>::min();

/**
 * @brief Sort a vector and drop duplicate entries in place.
 */
template <typename T>
inline void sortUnique(std::vector<T>& v) {
    std::ranges::sort(v);
    v.erase(std::ranges::unique(v).begin(), v.end());
}

}  // namespace DynamicTB::Preprocessing
