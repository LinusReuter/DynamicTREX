#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../../Helpers/Types.h"
#include "TransferTypes.h"

namespace DynamicTB::Preprocessing {

/**
 * @brief Sinks for the "level-0 affected event" set of Dynamic TREX.
 *
 * A stop event is level-0 affected *only if* its set of *minimized* incoming or outgoing
 * transfers changed during this update (an edge was added to, removed from, or flipped
 * into/out of the reduced set). That delta is the seed the bottom-up TREX customization
 * starts from.
 *
 * ### Why each mark carries a rank
 *
 * `CellTransferSearch::enqueue` relaxes an edge only while `rank >= level`, and a search is cut
 * off at its cell's boundary, so the searches that can ever touch an edge are those run for a
 * cell holding its source, at a level no higher than its rank. `ExactDecision::decideStop`
 * likewise only decides edges sourced in the cell it is handed. Marking an edge's source cell at
 * levels `[0, rank]` therefore covers exactly the work that edge can influence.
 *
 * An edge *entering* the reduced set enters at rank 0 (`TransferMeta{}`), and rank 0 is inert
 * above level 0 on both sides: `CellTransferSearch::enqueue` returns on `minLevel_ > rank`, and
 * `ExactDecision::decideStop` skips on `oldRank < levelRank`. Nothing above level 0 can see it
 * until a level-0 decision raises it, and that decision hands the new rank to `MarkCascade`. So
 * additions need bound 0 and nothing more.
 *
 * A departing edge needs its old rank, and that is worth spelling out, because the tempting
 * simplification -- give removals bound 0 too, since a removal that changes nothing at level 0
 * cannot matter higher -- is wrong, and wrong in the direction that loses journeys rather than the
 * one that merely wastes work. A removed edge is gone from the CSR, so `ExactDecision::decideStop`
 * never enumerates it and `MarkCascade` never fires for it. The only level-0 signal is some *other*
 * edge changing, and that not nessesary happen:
 *
 * > Level-0 cells `A` and `B` share a level-1 cell `C1`; `X` lies outside `C1`. Trip `S` runs
 * > `X -> A -> B`, trip `U` stays inside `A`, trip `T` has five stops, `t0..t3` in `B` and `t4` in
 * > `X`. The reduced set holds `P` (sourced in `A` on `S`, boarding `T` at index 2, rank 2), `W`
 * > (sourced in `A` on `S`, boarding `U` at index 0, rank 1) and `Q` (sourced on `U`, boarding `T`
 * > at index 0, rank 1).
 * >
 * > Only `S`'s entry into `A` crosses a border above level 0, so `C1` is searched from that single
 * > seed. There `P` reaches `t4` in `X` and escapes, so it is raised to 2 -- and relaxing it sets
 * > `DynamicReachedIndex` for `T` to 3, which clips `Q`'s label to `T[1,3)`, wholly inside `C1`.
 * > `Q` and `W` are not found, and stay at rank 1.
 * >
 * > Now `P` leaves the reduced set. Searching `A` at level 0, `Q`'s label widens to `T[1,5)` but is
 * > still cut at `t1`, which is in `B` and outside `A`, exactly as before: every level-0 decision
 * > is unchanged, so nothing cascades. Searching `C1` at level 1, nothing clips `T` any more, so
 * > `Q`'s label reaches `t4` in `X`, escapes, and unpacking it marks `Q` and -- through
 * > `parentTransfer` -- `W`. Both have to be raised to 2.
 * >
 * > With bound 2 the level-1 cell is marked and both are raised. With bound 0 only level 0 is
 * > marked, it reports no change, level 1 is never visited, and `Q` and `W` keep rank 1 against a
 * > true rank of 2, so a query prunes two transfers it still needs.
 *
 * What generalises from that: the deletion did not shrink the found set, it *moved* it. `T`'s
 * escaping segment was covered by `P`'s label and is now covered by `Q`'s. A deletion can create
 * marks, which is why "a removal can only ever decrement" does not hold.
 *
 * Sink concept (both directions of an edge are marked, because the customization needs
 * events whose incoming *or* outgoing reduced set moved):
 *
 *   void markEdgeChanged(PersistentStopEventId from, PersistentStopEventId to, uint8_t bound) noexcept;
 *   void markEvent(PersistentStopEventId event, uint8_t bound) noexcept;
 */

/**
 * @brief Bound for a disturbance whose pre-update rank cannot be recovered at the call site.
 */
inline constexpr std::uint8_t unknownRankBound = 0xFF;

/**
 * @brief One affected event and how far up the hierarchy its disturbance can matter.
 */
struct AffectedEvent {
    PersistentStopEventId event;
    std::uint8_t levelBound;
};

/**
 * @brief The no-op sink used when no TREX customization is attached.
 */
struct NullAffectedSink {
    inline void markEdgeChanged(PersistentStopEventId, PersistentStopEventId, std::uint8_t) noexcept {}
    inline void markEvent(PersistentStopEventId, std::uint8_t) noexcept {}
};

/**
 * @brief Per-thread accumulator for the affected set.
 *
 * One instance lives in each OpenMP thread's stack frame (exactly like the existing
 * TransferUpdateCounters), and is merged into the shared AffectedEvents under the same
 * `omp critical` that already merges the counters. Nothing here is thread-safe by itself.
 */
class AffectedEventCollector {
public:
    inline void markEdgeChanged(const PersistentStopEventId from, const PersistentStopEventId to,
                                const std::uint8_t levelBound) noexcept {
        events_.push_back({from, levelBound});
        events_.push_back({to, levelBound});
    }

    inline void markEvent(const PersistentStopEventId event, const std::uint8_t levelBound) noexcept {
        events_.push_back({event, levelBound});
    }

    inline void clear() noexcept { events_.clear(); }

    inline void reserve(const std::size_t n) { events_.reserve(n); }

    [[nodiscard]] inline const std::vector<AffectedEvent>& events() const noexcept { return events_; }

    inline std::vector<AffectedEvent>& events() noexcept { return events_; }

private:
    std::vector<AffectedEvent> events_;
};

/**
 * @brief The merged, deduplicated level-0 affected set of one update.
 */
class AffectedEvents {
public:
    inline void clear() noexcept { events_.clear(); }

    /**
     * @brief Merge one thread's accumulator. Call under the phase's `omp critical`.
     */
    inline void merge(const AffectedEventCollector& local) {
        const auto& src = local.events();
        events_.insert(events_.end(), src.begin(), src.end());
    }

    /**
     * @brief Sort and fold once, after all phases have merged.
     *
     * An event marked several times keeps the *highest* of its bounds.
     */
    inline void finalize() {
        std::sort(events_.begin(), events_.end(), [](const AffectedEvent& a, const AffectedEvent& b) noexcept {
            return (a.event != b.event) ? a.event < b.event : a.levelBound > b.levelBound;
        });
        const auto end = std::unique(events_.begin(), events_.end(),
                                     [](const AffectedEvent& a, const AffectedEvent& b) noexcept {
                                         return a.event == b.event;  // the highest bound sorts first
                                     });
        events_.erase(end, events_.end());
    }

    [[nodiscard]] inline const std::vector<AffectedEvent>& events() const noexcept { return events_; }

    [[nodiscard]] inline std::size_t size() const noexcept { return events_.size(); }

    [[nodiscard]] inline bool empty() const noexcept { return events_.empty(); }

private:
    std::vector<AffectedEvent> events_;
};

}  // namespace DynamicTB::Preprocessing
