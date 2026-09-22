#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../../../DataStructures/DynamicTimeTable/UpdateTypes.h"
#include "../../../Helpers/Assert.h"
#include "../../../Helpers/Vector/Vector.h"
#include "../preprocessing/AffectedEventSink.h"
#include "CellLayout.h"
#include "CustomizationInvariants.h"
#include "CustomizationTypes.h"

namespace DynamicTB::Customization {

/**
 * @brief Which cells the sweep has to visit, per level.
 *
 * One byte array per level, `2^(numberOfLevels - level)` entries -- under 128 KB in total at the
 * 16-level maximum. Allocated once for the process and cleared, never freed, per customization.
 */
class CellMarks {
public:
    /** Allocate for `levels` levels and clear. Reuses the existing buffer when the shape matches. */
    void reset(const int levels) {
        if (layout_.numberOfLevels() != levels) {
            layout_.reset(levels);
            marks_.assign(layout_.totalCells(), 0);
        } else {
            marks_.assign(marks_.size(), 0);
        }
    }

    int numberOfLevels() const noexcept { return layout_.numberOfLevels(); }

    std::size_t cellsAtLevel(const int level) const noexcept { return layout_.cellsAtLevel(level); }

    void mark(const int level, const CellId cell) noexcept { marks_[layout_.index(level, cell)] = 1; }

    /** Mark every cell of every level -- the full-customization configuration. */
    void markAll() noexcept { marks_.assign(marks_.size(), 1); }

    /**
     * Mark the cell containing `cell0` at every level up to and including `levelBound`.
     *
     * Cell ids are hierarchical bit prefixes, so the ancestor of a level-0 cell at level L is
     * just `cell0 >> L`.
     */
    void markUpTo(const CellId cell0, const int levelBound) noexcept {
        const int top = std::min(levelBound, numberOfLevels() - 1);
        for (int level = 0; level <= top; ++level) {
            mark(level, static_cast<CellId>(cell0 >> level));
        }
    }

    /** Append the marked cells of `level` to `out` (which is cleared first). */
    void collect(const int level, std::vector<CellId>& out) const {
        out.clear();
        const std::size_t base = layout_.index(level, 0);
        const std::size_t cells = cellsAtLevel(level);
        for (std::size_t c = 0; c < cells; ++c) {
            if (marks_[base + c] != 0) out.push_back(static_cast<CellId>(c));
        }
    }

    long long byteSize() const noexcept { return Vector::byteSize(marks_) + layout_.byteSize(); }

private:
    std::vector<std::uint8_t> marks_;
    LevelCellLayout layout_;
};

/**
 * @brief Seed policy for the full customization: mark every cell at every level.
 *
 * This is the reference result an incremental customization is checked against. It is
 * deliberately *not* a second implementation -- it is the same driver, the same kernel and the
 * same level rule, differing only in which cells end up in the worklist and in zeroing the ranks
 * first. One code path means a full-vs-incremental diff cannot be fooled by two implementations
 * drifting apart.
 *
 * The price of sharing the kernel is that a bug *inside* the kernel is invisible to that diff,
 * which is why the cross-check against the static TREX builder
 * (`compareDynamicCustomizationToStatic`) exits.
 */
struct FullCustomizationPolicy {
    /// The full customization starts from a blank slate; anything else would inherit stale ranks.
    static constexpr bool resetsRanks = true;
    /// Nothing to cascade into: every cell of every level is already marked.
    static constexpr bool cascades = false;

    void seed(const CellNetwork&, CellMarks& marks) { marks.markAll(); }
};


/**
 * @brief The no-op cascade, selected whenever the seed policy already marks everything.
 *
 * `NoCascade::raise` inlines to nothing, so a full-customization binary contains neither the
 * per-changed-edge cell lookup nor the marking loop.
 */
struct NoCascade {
    static constexpr bool enabled = false;
    explicit NoCascade(CellMarks&) noexcept {}
    void raise(CellId, int, int) const noexcept {}
};

/**
 * @brief Propagate a level-L rank decision to the levels above it.
 *
 * A transfer whose rank moved at level L can change what the searches of its cell's *ancestors*
 * discover, so those ancestors have to enter the worklist. The bound is `max(oldRank, newRank)`:
 * above that level the edge is not admitted on either side of the change, so it cannot influence
 * any decision there.
 */
class MarkCascade {
public:
    static constexpr bool enabled = true;

    explicit MarkCascade(CellMarks& marks) noexcept : marks_(&marks) {}

    void raise(const CellId cell0, const int decidedLevel, const int levelBound) const noexcept {
        const int top = std::min(levelBound, marks_->numberOfLevels() - 1);
        for (int level = decidedLevel + 1; level <= top; ++level) {
            marks_->mark(level, static_cast<CellId>(cell0 >> level));
        }
    }

private:
    CellMarks* marks_;
};

/**
 * @brief The incremental seed policy: mark only the cells one update can have disturbed.
 *
 * Two producers:
 *
 *  - **direct** -- every event whose set of *minimized* transfers changed during this update.
 *    That set is a by-product the minimization stage already computes. `AffectedEventCollector`
 *    accumulates it. Its cell is marked at every level up to the seed's `levelBound`. This carries
 *    more than "the edges of this cell changed": it is also the only cover for a *target* trip
 *    changing its FIFO group. That is safe only because a trip changes route exclusively
 *    through `extractTrip`, which lands it in `cancelledTrips`, whose cancellation phase clears its
 *    incoming edges and marks every source.
 *  - **cascade** -- see `MarkCascade`; it runs inside the level sweep rather than here.
 *
 * ### How high a seed is marked
 *
 * Each direct seed carries its own bound, produced where the disturbance happened: 0 for an edge
 * *entering* the reduced set, the departing edge's pre-update rank `r` for one *leaving* it. So an
 * addition is marked at level 0 only and the levels above it are reached by `MarkCascade` as ranks
 * actually move, while a removal is marked at levels `[0, r]` directly. Those are exactly the
 * passes that used to relax it.
 */
class IncrementalSeedPolicy {
public:
    /// The CSR carries the previous minute's ranks; the whole point is to keep the ones that hold.
    static constexpr bool resetsRanks = false;
    static constexpr bool cascades = true;

    /**
     * Bind this update's inputs.
     */
    void bind(const Preprocessing::AffectedEvents* affected,
              const DynamicTimeTable::ChangeSummary* changes) noexcept {
        affected_ = affected;
        changes_ = changes;
    }

    void seed(const CellNetwork& net, CellMarks& marks) {
        if constexpr (checkCustomizationInvariants) {
            std::string error;
            AssertMsg(tripOrder_.check(net, changes_, error), error);
        }

        directSeeds_ = 0;
        const int coarseBound = net.levels - 1;

        if (affected_ != nullptr) {
            // `AffectedEvents::finalize` already sorted the set and folded each event down to its
            // highest bound, so no dedup is needed here. Repeated marks would be harmless anyway --
            // a mark is an idempotent byte store -- which is why the cells themselves are never
            // deduplicated below.
            for (const Preprocessing::AffectedEvent& seed : affected_->events()) {
                const int bound = seed.levelBound == Preprocessing::unknownRankBound
                                      ? coarseBound
                                      : static_cast<int>(seed.levelBound);
                marks.markUpTo(cellOfPersistentEvent(net, seed.event), bound);
                ++directSeeds_;
            }
        }

        if constexpr (checkCustomizationInvariants) tripOrder_.snapshot(net);
    }

    std::size_t directSeeds() const noexcept { return directSeeds_; }

private:
    /**
     * The cell of a persistent event, whether or not it is still in the active timetable.
     */
    static CellId cellOfPersistentEvent(const CellNetwork& net, const PersistentStopEventId event) {
        const StopEventId flat = net.qd->persistentToFlatEvent[event];
        if (flat != noStopEvent) return net.cellOfEvent(flat);
        const DynamicTimeTable::PersistentStopEvent* stopEvent = net.data->getEvent(event);
        Ensure(stopEvent != nullptr, "Affected event " << event << " is not a minted stop event");
        return net.data->getCellIdOfStop(stopEvent->stop);
    }

    const Preprocessing::AffectedEvents* affected_ = nullptr;
    const DynamicTimeTable::ChangeSummary* changes_ = nullptr;
    std::size_t directSeeds_ = 0;
    TripFifoOrderWatch tripOrder_;
};

}  // namespace DynamicTB::Customization
