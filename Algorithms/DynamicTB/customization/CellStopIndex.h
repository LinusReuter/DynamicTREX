#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "../../../DataStructures/DynamicTimeTable/Data.h"
#include "../../../Helpers/Types.h"
#include "../../../Helpers/Vector/Vector.h"
#include "CellLayout.h"

namespace DynamicTB::Customization {

/**
 * @brief The stops of a (level, cell), as one contiguous range.
 *
 * `CellBorderIndex` answers "where does a route enter cell C" -- the seeds of the customization's
 * searches. This is the inverse direction, and it exists for the *decision* half: the exact rule
 * has to enumerate every transfer sourced inside C in order to decrement the ones the level's
 * searches did not find, and the only handle on "sourced inside C" is C's stops.
 */
class CellStopIndex {
public:
    CellStopIndex() = default;

    explicit CellStopIndex(const DynamicTimeTable::Data& data) { build(data); }

    /**
     * Rebuild from scratch by counting sort over the level-0 cell ids. Safe on an unpartitioned
     * network: the index then stays empty and every accessor returns an empty span.
     */
    void build(const DynamicTimeTable::Data& data) {
        stopsByCell_.clear();
        offset_.clear();
        layout_.reset(0);

        if (!data.hasPartition()) return;

        layout_.reset(data.getNumberOfLevels());
        const std::size_t numberOfStops = data.numberOfStops();
        const std::size_t cellsAtLevelZero = cellsAtLevel(0);

        // offset_[c+1] holds cell c's count first, then becomes its exclusive end after the
        // prefix sum -- the usual counting-sort two-pass, so nothing but the two arrays is
        // allocated.
        offset_.assign(cellsAtLevelZero + 1, 0);
        for (StopId stop(0); stop < StopId(numberOfStops); ++stop) {
            ++offset_[static_cast<std::size_t>(data.getCellIdOfStop(stop)) + 1];
        }
        for (std::size_t c = 1; c < offset_.size(); ++c) {
            offset_[c] += offset_[c - 1];
        }

        stopsByCell_.resize(numberOfStops);
        std::vector<std::uint32_t> cursor(offset_.begin(), offset_.end() - 1);
        for (StopId stop(0); stop < StopId(numberOfStops); ++stop) {
            stopsByCell_[cursor[data.getCellIdOfStop(stop)]++] = stop;
        }
    }

    /**
     * The stops of cell `cell` at `level`. `cell` is already shifted to the level's granularity,
     * i.e. `getCellIdOfStop(stop) >> level`, matching `CellBorderIndex::positions`.
     */
    std::span<const StopId> stops(const int level, const CellId cell) const noexcept {
        if (!layout_.contains(level, cell)) return {};
        const std::size_t lo = offset_[static_cast<std::size_t>(cell) << level];
        const std::size_t hi = offset_[static_cast<std::size_t>(cell + 1) << level];
        return {stopsByCell_.data() + lo, hi - lo};
    }

    int numberOfLevels() const noexcept { return layout_.numberOfLevels(); }

    std::size_t cellsAtLevel(const int level) const noexcept { return layout_.cellsAtLevel(level); }

    std::size_t stopCount() const noexcept { return stopsByCell_.size(); }

    /**
     * Cross-check against the timetable: every stop appears exactly once, and every filed stop's
     * cell id matches the bucket it sits in. The level-L accessor needs no separate check -- it
     * is a pure arithmetic consequence of the level-0 layout, which is what the second loop
     * verifies by re-deriving each level's ranges from the level-0 ones.
     *
     * O(stops * levels); intended for a shell command, not the hot path.
     */
    bool validate(const DynamicTimeTable::Data& data, std::string& error) const {
        if (!data.hasPartition()) return numberOfLevels() == 0;
        if (numberOfLevels() != data.getNumberOfLevels()) {
            error = "level count disagrees with the loaded partition";
            return false;
        }
        if (stopsByCell_.size() != data.numberOfStops()) {
            error = "index holds " + std::to_string(stopsByCell_.size()) + " stops, timetable has " +
                    std::to_string(data.numberOfStops());
            return false;
        }

        std::vector<bool> seen(data.numberOfStops(), false);
        for (std::size_t c = 0; c < cellsAtLevel(0); ++c) {
            for (const StopId stop : stops(0, static_cast<CellId>(c))) {
                if (static_cast<std::size_t>(stop) >= seen.size()) {
                    error = "filed stop is out of range";
                    return false;
                }
                if (seen[stop]) {
                    error = "stop " + std::to_string(static_cast<std::size_t>(stop)) + " is filed twice";
                    return false;
                }
                seen[stop] = true;
                if (data.getCellIdOfStop(stop) != c) {
                    error = "stop " + std::to_string(static_cast<std::size_t>(stop)) + " sits in the wrong bucket";
                    return false;
                }
            }
        }
        for (std::size_t i = 0; i < seen.size(); ++i) {
            if (!seen[i]) {
                error = "stop " + std::to_string(i) + " is missing from the index";
                return false;
            }
        }

        // Every coarser level must partition the same stops: a cell's span is the concatenation
        // of the spans of its two children.
        for (int level = 1; level < numberOfLevels(); ++level) {
            std::size_t total = 0;
            for (std::size_t c = 0; c < cellsAtLevel(level); ++c) {
                const auto parent = stops(level, static_cast<CellId>(c));
                const auto left = stops(level - 1, static_cast<CellId>(c << 1));
                const auto right = stops(level - 1, static_cast<CellId>((c << 1) | 1));
                if (parent.size() != left.size() + right.size() || (!parent.empty() && parent.data() != left.data())) {
                    error = "level " + std::to_string(level) + " cell " + std::to_string(c) +
                            " is not the concatenation of its children";
                    return false;
                }
                for (const StopId stop : parent) {
                    if (static_cast<std::size_t>(data.getCellIdOfStop(stop) >> level) != c) {
                        error = "stop leaks across a level-" + std::to_string(level) + " cell boundary";
                        return false;
                    }
                }
                total += parent.size();
            }
            if (total != stopsByCell_.size()) {
                error = "level " + std::to_string(level) + " does not cover every stop";
                return false;
            }
        }
        return true;
    }

    long long byteSize() const noexcept {
        return Vector::byteSize(stopsByCell_) + Vector::byteSize(offset_) + layout_.byteSize();
    }

private:
    std::vector<StopId> stopsByCell_;    // numberOfStops entries, sorted by level-0 cell id
    std::vector<std::uint32_t> offset_;  // 2^K + 1 entries; serves every level
    LevelCellLayout layout_;
};

}  // namespace DynamicTB::Customization
