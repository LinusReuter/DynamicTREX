#pragma once

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <type_traits>
#include <vector>

#include "../../../DataStructures/DynamicTimeTable/Data.h"
#include "../../../Helpers/Types.h"
#include "../../DynamicTimeTable/BuildQueryData.h"
#include "../../TripBased/Query/Types.h"
#include "../preprocessing/TransferTypes.h"
#include "CellBorderIndex.h"
#include "CellLayout.h"
#include "CellStopIndex.h"

namespace DynamicTB::Customization {

// Work counters are compile-time gated
#ifdef DYN_COLLECT_CUSTOMIZATION_STATS
inline constexpr bool collectCustomizationStats = true;
#else
inline constexpr bool collectCustomizationStats = false;
#endif

using RankChange = DynamicTB::Preprocessing::RankChange;

/**
 * @brief One level's decision for one edge, before the sweep's intermediates are folded away.
 *
 * `oldRank` is what makes the fold expressible ("did this edge end where it started?")
 */
struct PendingChange {
    PersistentStopEventId from;
    PersistentStopEventId to;
    std::uint8_t oldRank;
    std::uint8_t newRank;
};

/**
 * @brief An incoming border event: the trip leaves its current cell between `stopIndex` and
 * `stopIndex + 1`, so the cell-restricted search is seeded at `stopIndex + 1`.
 */
struct PackedIbe {
    std::uint32_t tripId : 24;
    std::uint32_t stopIndex : 8;

    constexpr PackedIbe(const TripId trip = noTripId, const StopIndex index = noStopIndex)
        : tripId(static_cast<std::uint32_t>(trip)), stopIndex(static_cast<std::uint32_t>(index)) {}

    constexpr TripId getTripId() const noexcept { return TripId(tripId); }
    constexpr StopIndex getStopIndex() const noexcept { return StopIndex(stopIndex); }
};

static_assert(sizeof(PackedIbe) == 4);
static_assert(std::is_trivially_copyable_v<PackedIbe>);

/**
 * @brief Everything the customization needs about the *current* network, as non-owning handles.
 */
struct CellNetwork {
    const DynamicTimeTable::Data* data = nullptr;
    const DynamicTimeTable::Algo::DynamicQueryData* qd = nullptr;
    const CellBorderIndex* borders = nullptr;
    const CellStopIndex* cellStops = nullptr;
    TripBased::Transfers* csr = nullptr;  // ranks live in labels[].getRank()/setRank()
    int levels = 0;

    bool valid() const noexcept {
        return data != nullptr && qd != nullptr && borders != nullptr && cellStops != nullptr && csr != nullptr &&
               levels > 0;
    }

    /// Cell of a flat stop event, derived rather than materialised, as this is expensive to build per update.
    CellId cellOfEvent(const StopEventId event) const noexcept {
        return data->getCellIdOfStop(qd->queryData.eventLookup[event].stop);
    }

    std::size_t edgeCount() const noexcept { return csr->labels.size(); }

    std::size_t flatEventCount() const noexcept { return qd->queryData.eventLookup.size(); }
};

/**
 * @brief Per-level work counters.
 */
struct LevelStats {
    std::uint64_t cellsProcessed = 0;
    std::uint64_t ibesRun = 0;
    std::uint64_t edgesEnumerated = 0;
    std::uint64_t edgesRaised = 0;
    std::uint64_t edgesDecremented = 0;
    std::uint64_t searchPhaseMicroseconds = 0;
    std::uint64_t decisionPhaseMicroseconds = 0;
};

/**
 * @brief The per-thread counting.
 */
struct NullLevelCounters {
    void enumerated() noexcept {}
    void raised() noexcept {}
    void decremented() noexcept {}
    void ibeRun() noexcept {}
    void reset() noexcept {}
    void mergeInto(LevelStats&) const noexcept {}
};

struct LevelCounters {
    LevelStats value{};

    void enumerated() noexcept { ++value.edgesEnumerated; }
    void raised() noexcept { ++value.edgesRaised; }
    void decremented() noexcept { ++value.edgesDecremented; }
    void ibeRun() noexcept { ++value.ibesRun; }
    void reset() noexcept { value = LevelStats{}; }

    void mergeInto(LevelStats& total) const noexcept {
        total.edgesEnumerated += value.edgesEnumerated;
        total.edgesRaised += value.edgesRaised;
        total.edgesDecremented += value.edgesDecremented;
        total.ibesRun += value.ibesRun;
    }
};

struct NullStats {
    static constexpr bool enabled = false;
    using Counters = NullLevelCounters;

    void resize(int) noexcept {}
    void clear() noexcept {}
    LevelStats& level(int) noexcept { return dummy_; }
    const std::vector<LevelStats>& levels() const noexcept { return empty_; }

private:
    LevelStats dummy_{};
    inline static const std::vector<LevelStats> empty_{};
};

struct CustomizationStats {
    static constexpr bool enabled = true;
    using Counters = LevelCounters;

    void resize(const int levels) { perLevel_.resize(static_cast<std::size_t>(levels)); }
    void clear() { perLevel_.assign(perLevel_.size(), LevelStats{}); }
    LevelStats& level(const int l) noexcept { return perLevel_[static_cast<std::size_t>(l)]; }
    const std::vector<LevelStats>& levels() const noexcept { return perLevel_; }

private:
    std::vector<LevelStats> perLevel_;
};

/// The stats type the build selected; the driver defaults to this.
using DefaultStats = std::conditional_t<collectCustomizationStats, CustomizationStats, NullStats>;

inline void printCustomizationStats(const std::vector<LevelStats>& perLevel, std::ostream& out) {
    if (perLevel.empty()) {
        out << "(build with -DDYN_COLLECT_CUSTOMIZATION_STATS for customization work counters)\n";
        return;
    }
    out << "Customization work per level\n";
    out << "  " << std::left << std::setw(6) << "level" << std::right << std::setw(10) << "cells" << std::setw(12)
        << "IBEs" << std::setw(14) << "edges enum" << std::setw(10) << "raised" << std::setw(10) << "decr"
        << std::setw(12) << "search us" << std::setw(12) << "decide us" << "\n";
    for (std::size_t l = 0; l < perLevel.size(); ++l) {
        const LevelStats& s = perLevel[l];
        out << "  " << std::left << std::setw(6) << l << std::right << std::setw(10) << s.cellsProcessed
            << std::setw(12) << s.ibesRun << std::setw(14) << s.edgesEnumerated << std::setw(10) << s.edgesRaised
            << std::setw(10) << s.edgesDecremented << std::setw(12) << s.searchPhaseMicroseconds << std::setw(12)
            << s.decisionPhaseMicroseconds << "\n";
    }
}

}  // namespace DynamicTB::Customization
