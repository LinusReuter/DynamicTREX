#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace DynamicTB::Customization {

/// A cell id already shifted to its level's granularity: `getCellIdOfStop(stop) >> level`.
using CellId = std::uint16_t;

/**
 * @brief The shape of the cell hierarchy: how many cells each level has, and where a level's
 * cells start in a flat per-(level, cell) array.
 */
class LevelCellLayout {
public:
    void reset(const int levels) {
        levels_ = levels;
        offset_.assign(static_cast<std::size_t>(levels) + 1, 0);
        std::size_t total = 0;
        for (int level = 0; level < levels; ++level) {
            offset_[static_cast<std::size_t>(level)] = total;
            total += cellsAtLevel(level);
        }
        offset_[static_cast<std::size_t>(levels)] = total;
    }

    int numberOfLevels() const noexcept { return levels_; }

    std::size_t cellsAtLevel(const int level) const noexcept { return std::size_t(1) << (levels_ - level); }

    std::size_t totalCells() const noexcept { return offset_.empty() ? 0 : offset_.back(); }

    bool contains(const int level, const CellId cell) const noexcept {
        return level >= 0 && level < levels_ && cell < cellsAtLevel(level);
    }

    std::size_t index(const int level, const CellId cell) const noexcept {
        return offset_[static_cast<std::size_t>(level)] + cell;
    }

    long long byteSize() const noexcept { return static_cast<long long>(sizeof(std::size_t) * offset_.size()); }

private:
    std::vector<std::size_t> offset_;
    int levels_ = 0;
};

}  // namespace DynamicTB::Customization
