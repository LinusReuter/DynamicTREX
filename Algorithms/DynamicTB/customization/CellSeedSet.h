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
 * Three producers:
 *
 *  - **direct** -- every event whose set of *minimized* transfers changed during this update.
 *    That set is a by-product the minimization stage already computes. `AffectedEventCollector`
 *    accumulates it. Its cell is marked at every level up to the seed's `levelBound`. This carries
 *    more than "the edges of this cell changed": it is also the only cover for a *target* trip
 *    changing its FIFO group. That is safe only because a trip changes route exclusively
 *    through `extractTrip`, which lands it in `cancelledTrips`, whose cancellation phase clears its
 *    incoming edges and marks every source.
 *  - **structural** -- every cell touched by the *old* route of a cancelled or extracted trip.
 *    Cancellation is the one disturbance the direct set cannot see, for two independent reasons:
 *    `TransferUpdate`'s cancellation phase clears a dying event's outgoing edges with no sink call
 *    at all, so the cell they were removed from gets no direct signal. Unchanged trips
 *    in that cell, which likewise produces no direct signal but may now need decrementing.
 *  - **cascade** -- see `MarkCascade`; it runs inside the level sweep rather than here.
 *
 * ### How high a seed is marked
 *
 * Each direct seed carries its own bound, produced by `markEdgeChanged` where the disturbance
 * happened: 0 for an edge *entering* the reduced set,
 * the departing edge's pre-update rank `r` for one *leaving* it. So an addition is marked at
 * level 0 only and the levels above it are reached by `MarkCascade` as ranks actually move, while
 * a removal is marked at levels `[0, r]` directly -- those are exactly the passes that used to
 * relax it. Collapsing the removal case to level 0 as well is *unsound*: a removed edge emits no
 * decision of its own, so nothing cascades, and the reached index's non-monotonicity means a
 * deletion can require a *raise* at level >= 1 while leaving every level-0 decision unchanged.
 * `markEdgeChanged` works that counterexample out in full.
 *
 * Two producers still fall back on the coarse `numberOfLevels - 1`, both because the pre-update
 * rank is not recoverable where they fire: `Preprocessing::unknownRankBound` (the mutator's
 * incoming removals, which have no metadata) and the structural seeds below. Over-marking is
 * safe, so these are a performance debt, not a correctness one.
 *
 * ### Why additions and in-place modifications seed nothing
 *
 * Both used to mark their whole route here, and both were removed once the full-customization diff
 * could confirm them redundant. The arguments are worth keeping, because they are what a future
 * change to the update pipeline would invalidate:
 *
 *  - **Additions.** A new incoming border event can mark something only if its seed trip has a
 *    *minimized* outgoing edge at an event inside the cell. They are marked during minimization
 *    as they are new
 *  - **In-place modifications.** A modification that survives `enforceFifo` keeps its route *and*
 *    its index in it; anything else is extracted and reappears as a cancellation plus an addition.
 *    Since the kernel reads no time, what is left cannot disturb a search at all -- only the
 *    transfer set it produced, which comes back through the direct set.
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
        structuralSeeds_ = 0;
        // The fallback for a disturbance whose reach is unknown: every level.
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

        if (changes_ != nullptr) {
            for (const auto& cancelled : changes_->cancelledTrips) {
                if (cancelled.oldRoute != noPersistentRouteId) {
                    markRoute(net, marks, cancelled.oldRoute, coarseBound);
                    continue;
                }
                // Fallback for an update that could not resolve the old route. Not coarser than
                // `markRoute`: `collectActiveEvents` snapshots the trip's whole effective stop
                // sequence, which is its route's stop sequence.
                for (const PersistentStopEventId event : cancelled.eventsOfCancelledTrips) {
                    marks.markUpTo(cellOfPersistentEvent(net, event), coarseBound);
                    ++structuralSeeds_;
                }
            }
            // `addedTrips` and `modifiedEvents` deliberately seed nothing
        }

        if constexpr (checkCustomizationInvariants) tripOrder_.snapshot(net);
    }

    std::size_t directSeeds() const noexcept { return directSeeds_; }
    std::size_t structuralSeeds() const noexcept { return structuralSeeds_; }

private:
    void markRoute(const CellNetwork& net, CellMarks& marks, const PersistentRouteId route, const int levelBound) {
        if (route == noPersistentRouteId) return;
        for (const StopId stop : net.data->getRoute(route).stopSequence) {
            marks.markUpTo(net.data->getCellIdOfStop(stop), levelBound);
            ++structuralSeeds_;
        }
    }

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
    std::size_t structuralSeeds_ = 0;
    TripFifoOrderWatch tripOrder_;
};

}  // namespace DynamicTB::Customization
