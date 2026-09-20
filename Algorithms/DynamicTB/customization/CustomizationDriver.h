#pragma once

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "../../../Helpers/Assert.h"
#include "../../../Helpers/Timer.h"
#include "../../../Helpers/Types.h"
#include "../../../Helpers/Vector/Vector.h"
#include "CellSeedSet.h"
#include "CellTransferSearch.h"
#include "CustomizationInvariants.h"
#include "CustomizationTypes.h"
#include "LevelDecision.h"

namespace DynamicTB::Customization {

/**
 * Whether a run's rank decisions are kept.
 */
enum class ChangeOutput { Collect, Discard };

/**
 * @brief The TREX customization: one level sweep over the exported, ranked CSR.
 *
 * ### Where it sits
 *
 * The per-minute pipeline already produces exactly the input this needs -- the reduced CSR, with
 * `TransferMeta::rank` riding in `EdgeLabel`'s spare bits -- so the customization mutates that
 * CSR in place and adds no export pass. The CSR is rebuilt from the store every minute, so
 * whatever the sweep decides has to be written back sparsely (`applyRankChanges`); the driver
 * hands out the change list and lets the caller decide when (and whether) to write it.
 *
 * ### A level pass is two phases: search, then decide
 *
 * ```
 * level L:  build worklist, expand IBEs
 *   parallel over IBEs   -- searches; mark the found set; touch no rank
 *   [barrier]
 *   parallel over stops  -- apply the rule; write ranks; emit changes
 *   [barrier]
 * ```
 */
template <class SeedPolicy = FullCustomizationPolicy, class Stats = DefaultStats>
class CustomizationDriver {
public:
    explicit CustomizationDriver(const int numberOfThreads = 1)
        : numberOfThreads_(std::max(1, numberOfThreads)), workspaces_(static_cast<std::size_t>(numberOfThreads_)) {}

    /**
     * Customize `net` in place. Ranks are read and written through `net.csr->labels`.
     * persistent store is not touched.
     *
     * `net.borders` must already have been synced against the current timetable
     * (`CellBorderIndex::syncNewRoutes`).
     */
    void run(const CellNetwork& net, const ChangeOutput output = ChangeOutput::Collect) {
        AssertMsg(net.valid(), "CellNetwork is incomplete");
        AssertMsg(net.borders->knownRouteCount() == net.data->routes().size(),
                  "CellBorderIndex has not been synced against the current timetable");

        if constexpr (checkCustomizationInvariants) {
            std::string error;
            AssertMsg(flatRoutesMirrorPersistentRoutes(net, error), error);
            AssertMsg(csrRowsAreSortedByPersistentTarget(net, error), error);
        }

        collectChanges_ = output == ChangeOutput::Collect;
        omp_set_num_threads(numberOfThreads_);
        stats_.resize(net.levels);
        stats_.clear();
        decision_.prepare(net);
        changes_.clear();

        pending_.clear();
        pending_.reserve(pendingHighWaterMark_);

#pragma omp parallel for schedule(static) if (numberOfThreads_ > 1)
        for (std::size_t w = 0; w < workspaces_.size(); ++w) {
            workspaces_[w].search.bind(net);
            workspaces_[w].changes.clear();
        }

        marks_.reset(net.levels);
        seeds_.seed(net, marks_);

        if constexpr (SeedPolicy::resetsRanks) {
            resetRanks(net);
        }

        for (int level = 0; level < net.levels; ++level) {
            runLevel(net, level);
            drainChanges();
        }

        mergeChanges();
    }

    /**
     * The rank decisions of the last `run`, in persistent (store) space. Feed to
     * `TransferUpdate::applyRankChanges` to make them survive the next export.
     */
    const std::vector<RankChange>& changes() const noexcept { return changes_; }

    const Stats& stats() const noexcept { return stats_; }

    /**
     * The seed policy, so a caller can hand it this update's inputs before `run`. The full
     * customization needs none; the incremental one is bound to the affected set and the change
     * summary of the update it is customizing.
     */
    SeedPolicy& seeds() noexcept { return seeds_; }
    const SeedPolicy& seeds() const noexcept { return seeds_; }

    long long scratchByteSize() const noexcept {
        long long total = decision_.byteSize() + marks_.byteSize();
        total += Vector::memoryUsageInBytes(ibes_) + Vector::memoryUsageInBytes(worklist_) +
                 Vector::memoryUsageInBytes(decisionStops_) + Vector::memoryUsageInBytes(pending_) +
                 Vector::memoryUsageInBytes(changes_);
        for (const auto& workspace : workspaces_) {
            total += workspace.search.byteSize() + Vector::memoryUsageInBytes(workspace.changes);
        }
        return total;
    }

private:
    struct Workspace {
        CellTransferSearch search;
        std::vector<PendingChange> changes;
        typename Stats::Counters counters;
    };

    /// `NoCascade` in a full customization: everything is marked, so there is nothing to raise.
    using Cascade = std::conditional_t<SeedPolicy::cascades, MarkCascade, NoCascade>;

    /**
     * Move the level's per-thread change buffers into the global list.
     *
     * Done per level, not once at the end, and that is load-bearing: an edge is decided at most
     * once per level but usually several times across the sweep (raised to 1 at level 0, to 2 at
     * level 1, ...). Draining at the end would interleave the levels by thread, so "the last
     * record for an edge" would no longer be its final rank.
     */
    void drainChanges() {
        for (auto& workspace : workspaces_) {
            if (collectChanges_) {
                pending_.insert(pending_.end(), workspace.changes.begin(), workspace.changes.end());
            }
            workspace.changes.clear();
        }
    }

    /**
     * Reduce the level-ordered change list to one net record per edge: the rank it started the
     * sweep with, and the rank it ended with.
     */
    void mergeChanges() {
        pendingHighWaterMark_ = std::max(pendingHighWaterMark_, pending_.size());
        if (pending_.empty()) return;
        std::stable_sort(pending_.begin(), pending_.end(), [](const PendingChange& a, const PendingChange& b) {
            return (a.from != b.from) ? a.from < b.from : a.to < b.to;
        });

        std::size_t i = 0;
        while (i < pending_.size()) {
            std::size_t j = i + 1;
            while (j < pending_.size() && pending_[j].from == pending_[i].from && pending_[j].to == pending_[i].to) {
                ++j;
            }
            const std::uint8_t oldRank = pending_[i].oldRank;
            const std::uint8_t newRank = pending_[j - 1].newRank;
            if (oldRank != newRank) {
                changes_.push_back(RankChange{pending_[i].from, pending_[i].to, newRank});
            }
            i = j;
        }
    }

    void resetRanks(const CellNetwork& net) {
        auto& labels = net.csr->labels;
        const std::size_t edges = labels.size();
#pragma omp parallel for schedule(static) if (numberOfThreads_ > 1)
        for (std::size_t edge = 0; edge < edges; ++edge) {
            labels[edge].setRank(0);
        }
    }

    void runLevel(const CellNetwork& net, const int level) {
        marks_.collect(level, worklist_);
        if (worklist_.empty()) return;  // nothing disturbed at this level; skip it entirely

        LevelStats& levelStats = stats_.level(level);
        if constexpr (Stats::enabled) levelStats.cellsProcessed = worklist_.size();

        expandIbes(net, level);

        // --- searches only, no rank is written, no change is emitted ----------------
        Timer phaseTimer;
        const std::size_t ibeCount = ibes_.size();
#pragma omp parallel if (numberOfThreads_ > 1)
        {
            Workspace& workspace = workspaces_[static_cast<std::size_t>(omp_get_thread_num())];
            auto sink = decision_.sink();
#pragma omp for schedule(dynamic, 32)
            for (std::size_t i = 0; i < ibeCount; ++i) {
                const PackedIbe ibe = ibes_[i];
                workspace.search.run(ibe.getTripId(), ibe.getStopIndex(), static_cast<std::uint8_t>(level), sink);
                workspace.counters.ibeRun();
            }
        }
        const double searchMicroseconds = phaseTimer.elapsedMicroseconds();

        // --- apply the rule ---------------------------------------------------------
        collectDecisionStops(net, level);
        const std::size_t stopCount = decisionStops_.size();
        // The cascade marks *later* levels, which are read only after this level's barrier, so the
        // decision loop may mark straight into the shared arrays.
        const Cascade cascade(marks_);
#pragma omp parallel if (numberOfThreads_ > 1)
        {
            Workspace& workspace = workspaces_[static_cast<std::size_t>(omp_get_thread_num())];
#pragma omp for schedule(dynamic, 64)
            for (std::size_t i = 0; i < stopCount; ++i) {
                decision_.decideStop(net, decisionStops_[i], level, workspace.changes, workspace.counters, cascade);
            }
        }
        const double decisionMicroseconds = phaseTimer.elapsedMicroseconds() - searchMicroseconds;

        if constexpr (Stats::enabled) {
            levelStats.searchPhaseMicroseconds = static_cast<std::uint64_t>(searchMicroseconds);
            levelStats.decisionPhaseMicroseconds = static_cast<std::uint64_t>(decisionMicroseconds);
            for (auto& workspace : workspaces_) {
                workspace.counters.mergeInto(levelStats);
                workspace.counters.reset();
            }
        }
    }

    /**
     * Flatten the marked cells' incoming border events into one list.
     */
    void expandIbes(const CellNetwork& net, const int level) {
        const auto& qd = net.qd->queryData;
        ibes_.clear();
        for (const CellId cell : worklist_) {
            for (const auto& position : net.borders->positions(level, cell)) {
                const RouteId route = net.qd->persistentToFlatRoute[position.route];
                if (route == noRouteId) continue;  // the route has no active trip this minute
                const TripId firstTrip = qd.firstTripOfRoute[route];
                const TripId endTrip = qd.firstTripOfRoute[route + 1];
                for (TripId trip = firstTrip; trip < endTrip; ++trip) {
                    // Every trip of a flat route shares its stop sequence, so a position derived
                    // from the persistent route is valid for all of them.
                    AssertMsg(static_cast<std::size_t>(position.index) + 1 < net.qd->numberOfStopsInTrip(trip),
                              "Border position runs past the end of a trip on its flat route");
                    ibes_.emplace_back(trip, position.index);
                }
            }
        }
    }

    void collectDecisionStops(const CellNetwork& net, const int level) {
        decisionStops_.clear();
        for (const CellId cell : worklist_) {
            const auto stops = net.cellStops->stops(level, cell);
            decisionStops_.insert(decisionStops_.end(), stops.begin(), stops.end());
        }
    }

    int numberOfThreads_;
    std::vector<Workspace> workspaces_;

    ExactDecision decision_{};
    SeedPolicy seeds_{};
    Stats stats_{};

    CellMarks marks_;
    std::vector<CellId> worklist_;
    std::vector<PackedIbe> ibes_;
    std::vector<StopId> decisionStops_;
    // Per-level decisions, still carrying `oldRank`; folded into `changes_` by `mergeChanges`.
    std::vector<PendingChange> pending_;
    std::size_t pendingHighWaterMark_ = 0;
    std::vector<RankChange> changes_;
    bool collectChanges_ = true;
};

/// Customize everything: every cell at every level, exact rule. The reference
/// result an incremental customization is diffed against.
using FullCustomizationDriver = CustomizationDriver<FullCustomizationPolicy, DefaultStats>;

/// Customize only what one update disturbed.
using IncrementalCustomizationDriver = CustomizationDriver<IncrementalSeedPolicy, DefaultStats>;

}  // namespace DynamicTB::Customization
