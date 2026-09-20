#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../../Helpers/Assert.h"
#include "../../../Helpers/Types.h"
#include "../../../Helpers/Vector/Vector.h"
#include "CustomizationTypes.h"

namespace DynamicTB::Customization {

/**
 * @brief `TripBased::TimestampedReachedIndex` against a `CellNetwork` instead of a `TripBased::Data`.
 *
 * Layout is identical to the static one: [ timestamp 16 | default label 8 | current label 8 ].
 * `rebuild` re-derives the defaults after a timetable update -- 4 bytes per flat trip, the one
 * network-proportional array the kernel keeps (and the one the static class also keeps).
 */
class DynamicReachedIndex {
public:
    void rebuild(const CellNetwork& net) {
        const auto& qd = net.qd->queryData;
        const std::size_t tripCount = qd.routeOfTrip.size();
        routeOfTrip_ = qd.routeOfTrip.data();
        firstTripOfRoute_ = qd.firstTripOfRoute.data();
        entries_.resize(tripCount);
        for (std::size_t trip = 0; trip < tripCount; ++trip) {
            const auto stops = static_cast<std::uint8_t>(qd.firstStopEventOfTrip[trip + 1] -
                                                         qd.firstStopEventOfTrip[trip]);
            entries_[trip] = pack(0, stops, stops);
        }
        timestamp_ = 0;
    }

    void clear() noexcept {
        timestamp_ = static_cast<std::uint16_t>((timestamp_ + 1) & 0xFFFFu);
        if (__builtin_expect(timestamp_ == 0, 0)) {
            for (auto& entry : entries_) {
                const std::uint8_t def = unpackDefault(entry);
                entry = pack(0, def, def);
            }
        }
    }

    StopIndex operator()(const TripId trip) noexcept { return StopIndex(getLabel(trip)); }

    bool alreadyReached(const TripId trip, const std::uint8_t index) noexcept { return getLabel(trip) <= index; }

    void update(const TripId trip, const StopIndex index) noexcept {
        // FIFO: reaching a trip at `index` also reaches every later trip of the same route there.
        const TripId routeEnd = firstTripOfRoute_[routeOfTrip_[trip] + 1];
        for (TripId i = trip; i < routeEnd; ++i) {
            if (getLabel(i) <= index) break;
            const std::uint8_t def = unpackDefault(entries_[i]);
            entries_[i] = pack(timestamp_, def, static_cast<std::uint8_t>(index));
        }
    }

    long long byteSize() const noexcept { return Vector::byteSize(entries_); }

private:
    static constexpr std::uint32_t pack(const std::uint16_t ts, const std::uint8_t def,
                                        const std::uint8_t cur) noexcept {
        return (static_cast<std::uint32_t>(ts) << 16) | (static_cast<std::uint32_t>(def) << 8) |
               static_cast<std::uint32_t>(cur);
    }
    static constexpr std::uint16_t unpackTimestamp(const std::uint32_t v) noexcept {
        return static_cast<std::uint16_t>(v >> 16);
    }
    static constexpr std::uint8_t unpackDefault(const std::uint32_t v) noexcept {
        return static_cast<std::uint8_t>((v >> 8) & 0xFFu);
    }
    static constexpr std::uint8_t unpackCurrent(const std::uint32_t v) noexcept {
        return static_cast<std::uint8_t>(v & 0xFFu);
    }

    std::uint8_t getLabel(const TripId trip) noexcept {
        const std::uint32_t v = entries_[trip];
        if (__builtin_expect(unpackTimestamp(v) != timestamp_, 0)) {
            const std::uint8_t def = unpackDefault(v);
            entries_[trip] = pack(timestamp_, def, def);
            return def;
        }
        return unpackCurrent(v);
    }

    std::vector<std::uint32_t> entries_;
    const RouteId* routeOfTrip_ = nullptr;
    const TripId* firstTripOfRoute_ = nullptr;
    std::uint16_t timestamp_ = 0;
};

/**
 * @brief The cell-restricted Event-TB search: one seed, one level, one cell.
 *
 * A direct port of the static `TripBased::TransferSearch` (`TransferSearchIBEs.h`) against a
 * `CellNetwork` and the exported CSR instead of a `TREXData` and its `stopEventGraph`.
 *
 * One object per thread per process, reused across cells, levels and minutes.
 */
class CellTransferSearch {
public:
    /**
     * Bind to a network. Cheap except for the reached index, which is O(flat trips); call once
     * per customization, not per search.
     */
    void bind(const CellNetwork& net) {
        net_ = &net;
        reachedIndex_.rebuild(net);
    }

    /**
     * Run the search seeded at the incoming border event `(trip, stopIndex + 1)`, restricted to
     * that event's cell at `level`. Every transfer on a path from the seed to an event that
     * leaves the cell is reported to `sink.mark(edge)`.
     */
    template <typename FoundSink>
    void run(const TripId trip, const StopIndex stopIndex, const std::uint8_t level, FoundSink& sink) {
        const auto& qd = *net_->qd;
        AssertMsg(stopIndex + 1 < qd.numberOfStopsInTrip(trip), "Seed runs past the end of its trip");

        clear();
        minLevel_ = level;
        currentCellId_ = net_->data->getCellIdOfStop(qd.getStop(trip, StopIndex(stopIndex + 1)));
        AssertMsg(currentCellId_ != net_->data->getCellIdOfStop(qd.getStop(trip, stopIndex)),
                  "Seed is not a border event: both sides sit in the same cell");

        enqueue(trip, StopIndex(stopIndex + 1));
        scanTrips();
        unpack(sink);
    }

    long long byteSize() const noexcept {
        return Vector::memoryUsageInBytes(queue_) + Vector::memoryUsageInBytes(edgeRanges_) +
               Vector::memoryUsageInBytes(toBeUnpacked_) + reachedIndex_.byteSize();
    }

private:
    static constexpr std::uint8_t MAX_ROUNDS = 16;

    struct TripLabel {
        StopEventId begin{noStopEvent};
        StopEventId end{noStopEvent};
        std::uint32_t parent{static_cast<std::uint32_t>(-1)};
        Edge parentTransfer{noEdge};
        bool unpacked{false};
    };

    struct EdgeRange {
        Edge begin{noEdge};
        Edge end{noEdge};
    };

    void clear() noexcept {
        queue_.clear();  // capacity retained: grows to a high-water mark, never shrinks
        toBeUnpacked_.clear();
        reachedIndex_.clear();
    }

    bool isEventInCell(const StopEventId event) const noexcept {
        return !((net_->cellOfEvent(event) ^ currentCellId_) >> minLevel_);
    }

    void scanTrips() noexcept {
        const auto& csr = *net_->csr;
        std::uint8_t round = 0;
        std::size_t roundBegin = 0;
        std::size_t roundEnd = queue_.size();

        while (roundBegin < roundEnd && round < MAX_ROUNDS) {
            ++round;

            // Pass 1: does this trip segment leave the cell? If so its journey has to be
            // unpacked, because the transfers on it are what let a query escape the cell.
            for (std::size_t i = roundBegin; i < roundEnd; ++i) {
                const TripLabel& label = queue_[i];
                bool isInSameCell = true;
                for (StopEventId j = label.begin; j < label.end; ++j) {
                    isInSameCell &= isEventInCell(j);
                }
                if (!isInSameCell) toBeUnpacked_.push_back(static_cast<std::uint32_t>(i));
            }

            // Pass 2: truncate each segment at its first out-of-cell event and resolve the CSR
            // range it relaxes. Separate from pass 3 so the edge ranges are computed before any
            // enqueue can grow the queue.
            // Grown to a high-water mark, never shrunk: a plain `resize` per round re-initialises
            // entries pass 2 overwrites immediately, once per round of every search.
            if (edgeRanges_.size() < queue_.size()) edgeRanges_.resize(queue_.size());
            for (std::size_t i = roundBegin; i < roundEnd; ++i) {
                TripLabel& label = queue_[i];
                for (StopEventId j = label.begin; j < label.end; ++j) {
                    if (!isEventInCell(j)) [[unlikely]] {
                        label.end = j;
                    }
                }
                edgeRanges_[i].begin = csr.beginOut[label.begin];
                edgeRanges_[i].end = csr.beginOut[label.end];
            }

            // Pass 3: relax. This is the loop whose order carries the exactness claim -- edges
            // are relaxed in CSR order, which is the store's `outgoing_sorted` order.
            for (std::size_t i = roundBegin; i < roundEnd; ++i) {
                const EdgeRange range = edgeRanges_[i];
                for (Edge edge = range.begin; edge < range.end; ++edge) {
                    enqueue(edge, static_cast<std::uint32_t>(i));
                }
            }

            roundBegin = roundEnd;
            roundEnd = queue_.size();
        }
    }

    void enqueue(const TripId trip, const StopIndex index) noexcept {
        if (reachedIndex_.alreadyReached(trip, static_cast<std::uint8_t>(index))) return;
        const StopEventId firstEvent = net_->qd->queryData.firstStopEventOfTrip[trip];
        TripLabel label;
        label.begin = StopEventId(firstEvent + index);
        label.end = StopEventId(firstEvent + reachedIndex_(trip));
        queue_.push_back(label);
        reachedIndex_.update(trip, index);
    }

    void enqueue(const Edge edge, const std::uint32_t parent) noexcept {
        const TripBased::EdgeLabel& label = net_->csr->labels[edge];
        // The admission filter. Note it is `>=`, not `==`: a transfer already known to be needed
        // at a coarser level is still part of this level's network.
        if (minLevel_ > label.getRank()) [[likely]]
            return;

        const std::uint8_t reachedTrip = static_cast<std::uint8_t>(reachedIndex_(label.getTrip()));
        if (reachedTrip <= static_cast<std::uint8_t>(label.getStopIndex())) [[likely]]
            return;

        TripLabel entry;
        entry.begin = label.getStopEvent();
        entry.end = StopEventId(label.getFirstEvent() + reachedTrip);
        entry.parent = parent;
        entry.parentTransfer = edge;
        queue_.push_back(entry);
        reachedIndex_.update(label.getTrip(), StopIndex(label.getStopIndex()));
    }

    template <typename FoundSink>
    void unpack(FoundSink& sink) {
        for (const std::uint32_t index : toBeUnpacked_) {
            unpackJourney(index, sink);
        }
    }

    template <typename FoundSink>
    void unpackJourney(std::uint32_t index, FoundSink& sink) {
        while (true) {
            TripLabel& label = queue_[index];
            if (label.parentTransfer == noEdge) break;  // reached the seed
            // Someone already walked this prefix in this search; everything above it is marked.
            if (label.unpacked) return;
            label.unpacked = true;
            sink.mark(label.parentTransfer);
            index = label.parent;
        }
        AssertMsg(index == 0, "A journey did not trace back to the incoming border event");
    }

    const CellNetwork* net_ = nullptr;
    std::vector<TripLabel> queue_;
    std::vector<EdgeRange> edgeRanges_;
    std::vector<std::uint32_t> toBeUnpacked_;
    DynamicReachedIndex reachedIndex_;
    std::uint8_t minLevel_ = 0;
    CellId currentCellId_ = 0;
};

}  // namespace DynamicTB::Customization
