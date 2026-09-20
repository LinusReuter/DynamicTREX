#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../../Helpers/Types.h"
#include "../../../Helpers/Vector/Vector.h"
#include "CustomizationTypes.h"

namespace DynamicTB::Customization {

/**
 * @brief Emit a rank decision for one CSR edge into a thread's change buffer.
 */
inline void emitRankChange(const CellNetwork& net, const Edge edge, const std::uint8_t oldRank,
                           const std::uint8_t newRank, const StopEventId flatFrom,
                           std::vector<PendingChange>& changes) {
    const StopEventId flatTo = StopEventId(net.csr->labels[edge].getStopEvent() - 1);
    changes.push_back({net.qd->flatToPersistentEvent[flatFrom], net.qd->flatToPersistentEvent[flatTo], oldRank,
                       newRank});
}

/**
 * @brief The exact rule: raise a transfer this level's searches found, lower one they did not.
 *
 * | condition | new rank |
 * |---|---|
 * | `rank < level` | unchanged (the edge was not admitted to this level's network) |
 * | `rank >= level`, found | `max(rank, level + 1)` |
 * | `rank >= level`, not found | `level` |
 *
 * The decrement is what makes this exact and what makes it expensive: proving a transfer is *no
 * longer* needed at a level requires enumerating every transfer sourced in the cell, not just
 * the ones the searches happened to touch. That enumeration is the decision phase's cost,
 * and it is the only reason `CellStopIndex` exists.
 */
class ExactDecision {
public:
    /**
     * Size the found-set array to the current export.
     */
    void prepare(const CellNetwork& net) {
        const std::size_t edges = net.edgeCount();
        if (foundEdges_.size() != edges) {
            foundEdges_.assign(edges, 0);
        }
    }

    struct Sink {
        std::uint8_t* foundEdges;
        void mark(const Edge edge) const noexcept { foundEdges[edge] = 1; }
    };

    Sink sink() noexcept { return Sink{foundEdges_.data()}; }

    /**
     * Decide every transfer sourced at one of this stop's events.
     */
    template <typename Counters, typename Cascade>
    void decideStop(const CellNetwork& net, const StopId stop, const int level, std::vector<PendingChange>& changes,
                    Counters& counters, const Cascade& cascade) {
        const auto& qd = net.qd->queryData;
        auto& labels = net.csr->labels;
        const std::uint8_t levelRank = static_cast<std::uint8_t>(level);
        // Every transfer sourced at this stop belongs to the stop's own cell, so one lookup covers
        // the whole loop. Only the cascade needs it, and `if constexpr` keeps it out of a full
        // customization entirely.
        CellId cell0 = 0;
        if constexpr (Cascade::enabled) cell0 = net.data->getCellIdOfStop(stop);

        for (const RAPTOR::RouteSegment& segment : qd.routesContainingStop(stop)) {
            const TripId firstTrip = qd.firstTripOfRoute[segment.routeId];
            const TripId endTrip = qd.firstTripOfRoute[segment.routeId + 1];
            for (TripId trip = firstTrip; trip < endTrip; ++trip) {
                const StopEventId event = StopEventId(qd.firstStopEventOfTrip[trip] + segment.stopIndex);
                const Edge begin = net.csr->beginOut[event];
                const Edge end = net.csr->beginOut[event + 1];
                for (Edge edge = begin; edge < end; ++edge) {
                    counters.enumerated();

                    const bool found = foundEdges_[edge] != 0;
                    foundEdges_[edge] = 0;

                    const std::uint8_t oldRank = labels[edge].getRank();
                    if (oldRank < levelRank) continue;  // not admitted at this level

                    const std::uint8_t newRank =
                        found ? std::max<std::uint8_t>(oldRank, static_cast<std::uint8_t>(levelRank + 1)) : levelRank;
                    if (newRank == oldRank) continue;

                    labels[edge].setRank(newRank);
                    if (newRank > oldRank) {
                        counters.raised();
                    } else {
                        counters.decremented();
                    }
                    // The change can only matter up to the higher of the two ranks
                    cascade.raise(cell0, level, std::max(oldRank, newRank));
                    emitRankChange(net, edge, oldRank, newRank, event, changes);
                }
            }
        }
    }

    long long byteSize() const noexcept { return Vector::byteSize(foundEdges_); }

private:
    std::vector<std::uint8_t> foundEdges_;
};

}  // namespace DynamicTB::Customization
