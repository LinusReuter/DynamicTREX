#pragma once

#include <cstddef>
#include <vector>

#include "../../../Helpers/Types.h"
#include "TransferTypes.h"

namespace DynamicTB::Preprocessing {

/**
 * @brief Sinks for the "level-0 affected event" set of Dynamic TREX.
 *
 * A stop event is level-0 affected iff its set of *minimized* incoming or outgoing
 * transfers changed during this update (an edge was added to, removed from, or flipped
 * into/out of the reduced set). That delta is the seed the bottom-up TREX customization
 * starts from.
 *
 * The sink is a **template parameter**, not a runtime flag: the marking calls sit in the
 * minimization inner loop, where a predictable-but-taken branch would cost more than the
 * work it guards. With NullAffectedSink every call below compiles to nothing.
 *
 * Sink concept (both directions of an edge are marked, because the customization needs
 * events whose incoming *or* outgoing reduced set moved):
 *
 *   void markEdgeChanged(PersistentStopEventId from, PersistentStopEventId to) noexcept;
 *   void markEvent(PersistentStopEventId event) noexcept;
 */

/**
 * @brief The no-op sink used when no TREX customization is attached.
 */
struct NullAffectedSink {
    inline void markEdgeChanged(PersistentStopEventId, PersistentStopEventId) noexcept {}
    inline void markEvent(PersistentStopEventId) noexcept {}
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
    inline void markEdgeChanged(const PersistentStopEventId from, const PersistentStopEventId to) noexcept {
        events_.push_back(from);
        events_.push_back(to);
    }

    inline void markEvent(const PersistentStopEventId event) noexcept { events_.push_back(event); }

    inline void clear() noexcept { events_.clear(); }

    inline void reserve(const std::size_t n) { events_.reserve(n); }

    [[nodiscard]] inline const std::vector<PersistentStopEventId>& events() const noexcept { return events_; }

    inline std::vector<PersistentStopEventId>& events() noexcept { return events_; }

private:
    std::vector<PersistentStopEventId> events_;
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
     * @brief Sort + deduplicate once, after all phases have merged.
     */
    inline void finalize() { sortUnique(events_); }

    [[nodiscard]] inline const std::vector<PersistentStopEventId>& events() const noexcept { return events_; }

    [[nodiscard]] inline std::size_t size() const noexcept { return events_.size(); }

    [[nodiscard]] inline bool empty() const noexcept { return events_.empty(); }

private:
    std::vector<PersistentStopEventId> events_;
};

}  // namespace DynamicTB::Preprocessing
