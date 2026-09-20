#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "../../../DataStructures/DynamicTimeTable/Data.h"
#include "../../../Helpers/Vector/Vector.h"
#include "../../../Helpers/Types.h"
#include "CellLayout.h"

namespace DynamicTB::Customization {

/**
 * @brief Where routes cross cell borders, indexed by (level, cell).
 *
 * The TREX customization seeds one cell-restricted search per *incoming border event*:
 * a stop event from which the trip immediately leaves the current cell. The static builder
 * recomputes that seed list from scratch for every level (`BuilderIBEs::collectAllIBEsOnLowestLevel`
 * plus `filterIrrelevantIBEs`), which is affordable once but not once per minute.
 *
 * The geometry, however, is invariant: RT updates never add stops and never repartition, and a
 * persistent route's stop sequence is immutable by construction. So *where* the borders are is
 * fixed; only which trips currently sit on those routes changes. This index computes the
 * positions once and hands them out as spans, leaving the (cheap) expansion over the route's
 * current trips to the caller -- which is also what keeps this correct when trips come and go.
 *
 * Level convention matches the static side: level 0 is the finest partition, and "same cell at
 * level L" is `!((a ^ b) >> L)`. A position whose crossing mask is `m = cell(a) ^ cell(b)` is a
 * border at exactly the levels `[0, bit_width(m))`.
 */
class CellBorderIndex {
public:
    /**
     * A route position at which the trip crosses a cell border: the crossing is
     * `index -> index + 1`, so `index` is the last stop still inside the source cell. This
     * matches the static `PackedIBE` convention in `BuilderIBEs.h`, where the search is then
     * seeded at `index + 1`.
     */
    struct BorderPosition {
        PersistentRouteId route;
        StopIndex index;
    };

    static_assert(sizeof(BorderPosition) == 8, "BorderPosition should stay pointer-sized");

    CellBorderIndex() = default;

    explicit CellBorderIndex(const DynamicTimeTable::Data& data) { build(data); }

    /**
     * Rebuild from scratch. Safe to call on an unpartitioned network: the index then stays
     * empty and `numberOfLevels()` is 0, so every accessor returns an empty span.
     */
    void build(const DynamicTimeTable::Data& data) {
        buckets_.clear();
        layout_.reset(0);
        positionCount_ = 0;

        if (!data.hasPartition()) return;

        // One bucket per (level, cell). Level L has 2^(numberOfLevels - L) cells, so the total is
        // 2^(numberOfLevels+1) - 2 buckets -- ~130k empty vectors (3 MB) at the 16-level maximum,
        // which is noise next to the transfer store and buys O(1) lookup plus cheap appends.
        layout_.reset(data.getNumberOfLevels());
        buckets_.resize(layout_.totalCells());

        const auto& routes = data.routes();
        for (std::size_t r = 0; r < routes.size(); ++r) {
            insertRoute(data, PersistentRouteId(r));
        }
        knownRouteCount_ = routes.size();
    }

    /**
     * Register every route minted since the last build/sync.
     *
     * `Data::routes_` is append-only and persistent route ids are dense and monotone, so "new"
     * is just "id >= the count we last saw".
     */
    void syncNewRoutes(const DynamicTimeTable::Data& data) {
        if (numberOfLevels() == 0) return;
        const std::size_t routeCount = data.routes().size();
        for (std::size_t r = knownRouteCount_; r < routeCount; ++r) {
            insertRoute(data, PersistentRouteId(r));
        }
        knownRouteCount_ = routeCount;
    }

    /** Routes the index has already filed; `syncNewRoutes` picks up everything beyond this. */
    std::size_t knownRouteCount() const noexcept { return knownRouteCount_; }

    /**
     * Positions crossing into `cell` at `level`. `cell` is a cell id already shifted to the
     * level's granularity, i.e. `getCellIdOfStop(stop) >> level`.
     */
    std::span<const BorderPosition> positions(const int level, const CellId cell) const noexcept {
        if (!layout_.contains(level, cell)) return {};
        return buckets_[layout_.index(level, cell)];
    }

    int numberOfLevels() const noexcept { return layout_.numberOfLevels(); }

    std::size_t cellsAtLevel(const int level) const noexcept { return layout_.cellsAtLevel(level); }

    /** Total filed positions, counting a position once per level it is a border at. */
    std::size_t positionCount() const noexcept { return positionCount_; }

    /**
     * Cross-check the index against a direct re-derivation from the timetable.
     *
     * O(border positions); intended for a shell command on a real partition, not the hot path.
     */
    bool validate(const DynamicTimeTable::Data& data, std::string& error) const {
        if (!data.hasPartition()) return numberOfLevels() == 0;
        if (numberOfLevels() != data.getNumberOfLevels()) {
            error = "level count disagrees with the loaded partition";
            return false;
        }
        // Anything past this count was minted by an update and never filed -- see syncNewRoutes.
        if (knownRouteCount_ != data.routes().size()) {
            error = "index knows " + std::to_string(knownRouteCount_) + " routes, timetable has " +
                    std::to_string(data.routes().size()) + " (syncNewRoutes was not called)";
            return false;
        }

        std::size_t filed = 0;
        for (int level = 0; level < numberOfLevels(); ++level) {
            // std::size_t, not uint16_t: at 16 levels the finest level has 65536 cells and a
            // 16-bit counter would wrap instead of terminating.
            for (std::size_t c = 0; c < cellsAtLevel(level); ++c) {
                const CellId cell = static_cast<CellId>(c);
                for (const BorderPosition& position : positions(level, cell)) {
                    const auto& stopSequence = data.getRoute(position.route).stopSequence;
                    const std::size_t i = static_cast<std::size_t>(position.index);
                    if (i + 1 >= stopSequence.size()) {
                        error = "position points past the end of its route";
                        return false;
                    }
                    const std::uint16_t from = data.getCellIdOfStop(stopSequence[i]);
                    const std::uint16_t to = data.getCellIdOfStop(stopSequence[i + 1]);
                    if (!((from ^ to) >> level)) {
                        error = "position filed at a level it does not cross at";
                        return false;
                    }
                    if ((to >> level) != cell) {
                        error = "position filed under the wrong cell";
                        return false;
                    }
                    ++filed;
                }
            }
        }
        if (filed != positionCount_) {
            error = "bucket contents disagree with the position count";
            return false;
        }

        // Completeness: recount the crossings straight off the timetable.
        std::size_t expected = 0;
        for (std::size_t r = 0; r < data.routes().size(); ++r) {
            const auto& stopSequence = data.routes()[PersistentRouteId(r)].stopSequence;
            for (std::size_t i = 1; i < stopSequence.size(); ++i) {
                const std::uint16_t mask = static_cast<std::uint16_t>(data.getCellIdOfStop(stopSequence[i - 1]) ^
                                                                     data.getCellIdOfStop(stopSequence[i]));
                expected += std::min(static_cast<int>(std::bit_width(mask)), numberOfLevels());
            }
        }
        if (filed != expected) {
            error = "index holds " + std::to_string(filed) + " positions, timetable has " + std::to_string(expected);
            return false;
        }
        return true;
    }

    long long byteSize() const noexcept { return layout_.byteSize() + Vector::byteSize(buckets_); }

private:
    void insertRoute(const DynamicTimeTable::Data& data, const PersistentRouteId route) {
        const auto& stopSequence = data.getRoute(route).stopSequence;
        if (stopSequence.size() < 2) return;

        CellId previousCell = data.getCellIdOfStop(stopSequence[0]);
        for (std::size_t i = 1; i < stopSequence.size(); ++i) {
            const CellId currentCell = data.getCellIdOfStop(stopSequence[i]);
            const CellId crossingMask = static_cast<CellId>(previousCell ^ currentCell);
            previousCell = currentCell;
            if (crossingMask == 0) continue;

            // `mask >> level` is non-zero exactly for level < bit_width(mask), so the levels at
            // which this is a border form a prefix. Clamped because a cell id may in principle
            // carry bits above the derived level count.
            const int maxLevel = std::min(static_cast<int>(std::bit_width(crossingMask)), numberOfLevels());
            const BorderPosition position{route, StopIndex(i - 1)};
            for (int level = 0; level < maxLevel; ++level) {
                buckets_[layout_.index(level, static_cast<CellId>(currentCell >> level))].push_back(position);
                ++positionCount_;
            }
        }
    }

    // One vector per (level, cell), indexed through `layout_`.
    std::vector<std::vector<BorderPosition>> buckets_;
    LevelCellLayout layout_;
    std::size_t positionCount_ = 0;
    // The only mutable state the index carries across updates: how far into the (append-only)
    // persistent route array it has already filed positions.
    std::size_t knownRouteCount_ = 0;
};

}  // namespace DynamicTB::Customization
