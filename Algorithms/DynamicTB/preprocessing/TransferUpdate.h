#pragma once

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../../DataStructures/DynamicTimeTable/UpdateTypes.h"
#include "../../../Helpers/MultiThreading.h"
#include "../../../Helpers/PhaseTimings.h"
#include "../../../Helpers/Timer.h"
#include "../../../Helpers/Types.h"
#include "../../../Helpers/UpdateCounters.h"
#include "../../DynamicTimeTable/BuildQueryData.h"
#include "AffectedEventSink.h"
#include "TransferDiscovery.h"
#include "TransferExport.h"
#include "TransferMinimizer.h"
#include "TransferStoreMutator.h"
#include "TransferTypes.h"

namespace DynamicTB::Preprocessing {

/**
 * @class TransferUpdate
 * @brief Phase driver for dynamic transfer updates driven by DynamicTimeTable::ChangeSummary.
 *
 * - Nodes: PersistentStopEventId (0..n-1)
 * - Edges: transfer from source event -> target event
 * - Uses DynamicQueryData for all lookups.
 * - Store contains the FULL set; minimization only toggles TransferMeta::isMinimized.
 *
 * The actual work lives in four collaborators:
 *   TransferDiscovery    -- what transfers exist, from the timetable alone
 *   TransferStoreMutator -- the only code that mutates the store
 *   TransferMinimizer    -- the reduced-set kernel
 *   TransferExporter     -- persistent store -> flat (ranked) CSR
 *
 * Template parameters:
 *   Store        the concrete transfer store; templated (rather than the ITransferStore
 *                base) so per-edge store calls inline instead of dispatching virtually.
 *   AffectedSink where the level-0 affected-event delta goes. NullAffectedSink compiles the
 *                whole collection away, which is what plain Dynamic TB runs use.
 */
template <class Store, class AffectedSink = NullAffectedSink>
class TransferUpdate {
public:
    using DynamicQueryData = DynamicTimeTable::Algo::DynamicQueryData;
    using NodeID = PersistentStopEventId;
    using Mutator = TransferStoreMutator<Store, AffectedSink>;
    using Minimizer = TransferMinimizer<Store, AffectedSink>;
    using Exporter = TransferExporter<Store>;

    static constexpr bool collectsAffected = !std::is_same_v<AffectedSink, NullAffectedSink>;


    using MinTarget = ::DynamicTB::Preprocessing::MinTarget;
    static constexpr int noTimeCutoff = ::DynamicTB::Preprocessing::noTimeCutoff;

    static_assert(sizeof(typename Store::OutEdge) == 8,
                  "TransferMeta should be small");

    explicit TransferUpdate(Store& store) : store_(store) {}

    /**
     * @brief The level-0 affected-event set produced by the most recent applyFullUpdates().
     * Always empty when AffectedSink is NullAffectedSink.
     */
    [[nodiscard]] inline const AffectedEvents& affectedEvents() const noexcept { return affected_; }

    /**
     * @brief Full rebuild: clears the store and (re)discovers all outgoing transfers.
     */
    void buildInitialFullTransfers(const DynamicQueryData& queryData, const int numberOfThreads = 1) {
        queryData_ = &queryData;

        // Total number of persistent stop events
        const std::size_t eventCount = queryData_->persistentToFlatEvent.size();

        if (eventCount == 0) {
            // Clear any previous data, if present.
            store_.clear();
            return;
        }

        // 1) Clear store (full clear; incoming will be rebuilt in bulk)
        store_.clear();

        // 2) Prepare outgoing-only init mode
        const NodeID maxEventId = NodeID(eventCount - 1);
        store_.begin_outgoing_init(maxEventId);

        // 3) Discover all outgoing transfers (no diff; init-only add). Each iteration
        // writes only to its own event's outgoing slot (pre-sized by begin_outgoing_init),
        // so this is safe to parallelize without locking.
        const TransferDiscovery discovery(queryData);
        const int threads = std::max(1, numberOfThreads);
        omp_set_num_threads(threads);
#pragma omp parallel if (threads > 1)
        {
            if (omp_get_num_threads() > 1) pinThreadToCoreId(omp_get_thread_num() % numberOfCores());
            DiscoveryWorkspace ws;
#pragma omp for schedule(dynamic, 1024)
            for (std::size_t i = 0; i < eventCount; ++i) {
                PersistentStopEventId event(i);

                // Skip invalid/removed events
                StopEventId flatEvent = queryData_->persistentToFlatEvent[i];
                if (flatEvent == noStopEvent) continue;

                ws.desired.clear();
                discovery.computeOutgoingTransfers(flatEvent, ws, ws.desired);
                if (!ws.desired.empty()) {
                    store_.reserve_outgoing(event, ws.desired.size());
                    store_.add_outgoing_edges_init(event, ws.desired, TransferMeta{});
                }
            }
        }

        // 4) Rebuild incoming / sync_barrier()
        store_.finish_outgoing_init();
    }

    /**
     * @brief Incremental FULL-set update pipeline driven by the latest ChangeSummary.
     * Evaluates cancellation, outgoing, and incoming discovery phases safely in parallel.
     */
    void applyFullUpdates(const DynamicTimeTable::ChangeSummary& changes, const DynamicQueryData& queryData,
                          const int numberOfThreads, const int nowSeconds = noTimeCutoff,
                          PhaseTimings* outPhases = nullptr) {
        Timer baseTimer;
        const int threads = std::max(1, numberOfThreads);
        omp_set_num_threads(threads);
        queryData_ = &queryData;
        affected_.clear();
        if constexpr (collectTransferStats) stats_counters_ = {};
        const std::size_t eventCount = queryData_->persistentToFlatEvent.size();
        if (eventCount == 0) {
            store_.clear();
            return;
        }

        const NodeID maxEventId = NodeID(eventCount - 1);
        store_.add_nodes(maxEventId);

        const TransferDiscovery discovery(queryData);
        const Mutator mutator(store_, queryData);

        std::vector<PersistentStopEventId>& toDiscoverOutgoing = toDiscoverOutgoing_;
        std::vector<PersistentStopEventId>& toDiscoverIncoming = toDiscoverIncoming_;
        // Trips whose minimization must be re-run, kept in flat space (stable for this call).
        std::vector<MinTarget>& globalTripsToMinimize = globalTripsToMinimize_;
        toDiscoverOutgoing.clear();
        toDiscoverIncoming.clear();
        globalTripsToMinimize.clear();

        store_.allowTemporaryInconsistent(true);

        // Phase 1: Parallel Cancellations
#pragma omp parallel if (threads > 1)
        {
            if (omp_get_num_threads() > 1) pinThreadToCoreId(omp_get_thread_num() % numberOfCores());
            std::vector<MinTarget> localTrips;
            std::vector<std::pair<NodeID, TransferMeta>> removedIncoming;
            TransferUpdateCounters lc;
            AffectedSink sink;
#pragma omp for schedule(dynamic, 16)
            for (const auto& cancelledTrip : changes.cancelledTrips) {
                if constexpr (collectTransferStats) ++lc.cancelledTripsProcessed;
                for (const auto event : cancelledTrip.eventsOfCancelledTrips) {
                    if constexpr (collectTransferStats) lc.outgoingEdgesCleared += store_.outgoing_unsorted(event).size();
                    store_.clear_outgoing(event);
                    // Sources lose their outgoing edge into this cancelled event. Only a
                    // source whose edge was MINIMIZED can have its reduction change; a
                    // non-minimized edge writes no StopLabels during reduction, so its
                    // removal leaves every other keep-flag untouched.
                    removedIncoming.clear();
                    store_.clear_incoming_with_meta(event, removedIncoming);
                    if constexpr (collectTransferStats) lc.incomingEdgesCleared += removedIncoming.size();
                    for (const auto& [from, meta] : removedIncoming) {
                        if (meta.isMinimized) {
                            mutator.recordSourceTripOfEvent(from, localTrips);
                            // The source lost a reduced-set edge => level-0 affected. The
                            // target is going away with its trip, so only the source matters.
                            sink.markEvent(from);
                        }
                    }
                }
            }
#pragma omp critical
            {
                globalTripsToMinimize.insert(globalTripsToMinimize.end(), localTrips.begin(), localTrips.end());
                mergeSink(sink);
                if constexpr (collectTransferStats) stats_counters_ += lc;
            }
        }
        store_.sync_barrier();

        collectDiscoveryTargets(changes, toDiscoverOutgoing, toDiscoverIncoming);
        if constexpr (collectTransferStats) {
            stats_counters_.discoverOutgoingEvents += toDiscoverOutgoing.size();
            stats_counters_.discoverIncomingEvents += toDiscoverIncoming.size();
        }

        // Phase 2: Parallel Outgoing Discovery
#pragma omp parallel if (threads > 1)
        {
            if (omp_get_num_threads() > 1) pinThreadToCoreId(omp_get_thread_num() % numberOfCores());
            std::vector<MinTarget> localTrips;
            TransferUpdateCounters lc;
            AffectedSink sink;
            DiscoveryWorkspace ws;
#pragma omp for schedule(dynamic, 16)
            for (const auto i : toDiscoverOutgoing) {
                updateOutgoingForEvent(discovery, mutator, i, ws, localTrips, lc, sink);
            }
#pragma omp critical
            {
                globalTripsToMinimize.insert(globalTripsToMinimize.end(), localTrips.begin(), localTrips.end());
                mergeSink(sink);
                if constexpr (collectTransferStats) stats_counters_ += lc;
            }
        }
        store_.sync_barrier();

        // Phase 3: Parallel Incoming Discovery
        // Domination cleanups are recorded, not executed here: they mutate shared
        // source outgoing lists and must run sequentially after.
        std::vector<PendingDominationCleanup>& globalPendingCleanups = globalPendingCleanups_;
        globalPendingCleanups.clear();
#pragma omp parallel if (threads > 1)
        {
            if (omp_get_num_threads() > 1) pinThreadToCoreId(omp_get_thread_num() % numberOfCores());
            std::vector<MinTarget> localTrips;
            std::vector<PendingDominationCleanup> localCleanups;
            std::vector<PersistentStopEventId> newlyInserted;
            TransferUpdateCounters lc;
            AffectedSink sink;
            DiscoveryWorkspace ws;
#pragma omp for schedule(dynamic, 16)
            for (const auto i : toDiscoverIncoming) {
                updateIncomingForEvent(discovery, mutator, i, ws, localTrips, localCleanups, newlyInserted, lc, sink);
            }
#pragma omp critical
            {
                globalTripsToMinimize.insert(globalTripsToMinimize.end(), localTrips.begin(), localTrips.end());
                globalPendingCleanups.insert(globalPendingCleanups.end(), localCleanups.begin(), localCleanups.end());
                mergeSink(sink);
                if constexpr (collectTransferStats) stats_counters_ += lc;
            }
        }
        store_.sync_barrier();

        {
            AffectedSink sink;
            for (const auto& [fromEvent, flatToEvent] : globalPendingCleanups) {
                if constexpr (collectTransferStats) ++stats_counters_.dominationCleanups;
                mutator.dominationCleanupForInsertedTransfer(fromEvent, flatToEvent, globalTripsToMinimize,
                                                             stats_counters_, sink);
            }
            mergeSink(sink);
        }

        store_.allowTemporaryInconsistent(false);

        // Phase 4: Parallel Processing for Changed Arrivals (later OR earlier).
        // Two effects of a changed arrival on trip T:
        //  (a) SELF: T's own minimization seeds StopLabels from the arrival times of all
        //      of T's stops and accumulates them while scanning stops backward, so a
        //      changed arrival at any stop affects the keep-decisions of EARLIER stops of
        //      T. => T itself must be re-minimized.
        //  (b) UPSTREAM: sources boarding T at stop index j need re-minimization only if
        //      some arrival CHANGED at a stop index > j. maxChangedIndex is the largest
        //      such index, so incoming edges into events at index >= maxChangedIndex are
        //      unaffected and skipped. A trip's flat stop events are laid out in stop-index order.
#pragma omp parallel if (threads > 1)
        {
            if (omp_get_num_threads() > 1) pinThreadToCoreId(omp_get_thread_num() % numberOfCores());
            std::vector<MinTarget> localTrips;
            TransferUpdateCounters lc;
#pragma omp for schedule(dynamic, 1024)
            for (const auto& [changedTrip, maxChangedIndex] : changes.tripsWithChangedArrivals) {
                const TripId flatTrip = queryData_->persistentToFlatTrip[changedTrip];
                if (flatTrip == noTripId) continue;
                const auto& qd = queryData_->queryData;
                const StopEventId firstEvent = qd.firstStopEventOfTrip[flatTrip];
                const std::size_t numStops = qd.firstStopEventOfTrip[flatTrip + 1] - firstEvent;
                if (numStops == 0) continue;
                if constexpr (collectTransferStats) ++lc.arrivalPropagationTrips;

                // (a) SELF: re-minimize the changed trip itself. Its arrivals changed up to
                // maxChangedIndex, so keep-decisions can differ from there down.
                localTrips.emplace_back(flatTrip, maxChangedIndex);

                // (b) UPSTREAM: re-minimize sources feeding stops before the last change.
                const std::size_t limit = std::min(static_cast<std::size_t>(maxChangedIndex) + 1, numStops);
                for (std::size_t idx = 0; idx < limit; ++idx) {
                    const PersistentStopEventId pEvent =
                        queryData_->flatToPersistentEvent[StopEventId(firstEvent + idx)];
                    for (const auto from : store_.incoming_sorted(pEvent)) {
                        if constexpr (collectTransferStats) ++lc.arrivalUpstreamSources;
                        mutator.recordSourceTripOfEvent(from, localTrips);
                    }
                }
            }
#pragma omp critical
            {
                globalTripsToMinimize.insert(globalTripsToMinimize.end(), localTrips.begin(), localTrips.end());
                if constexpr (collectTransferStats) stats_counters_ += lc;
            }
        }

        aggregateMaxByTrip(globalTripsToMinimize);

        if (outPhases != nullptr) {
            outPhases->baseTransferUpdate +=
                std::chrono::microseconds(static_cast<long long>(baseTimer.elapsedMicroseconds()));
        }

        Timer minimizationTimer;
        if constexpr (collectTransferStats) stats_counters_.tripsMinimized += globalTripsToMinimize.size();
        updateMinimizedTransfers(globalTripsToMinimize, queryData, numberOfThreads, nowSeconds, &stats_counters_);
        if (outPhases != nullptr) {
            outPhases->minimizationUpdate +=
                std::chrono::microseconds(static_cast<long long>(minimizationTimer.elapsedMicroseconds()));
        }

        if constexpr (collectsAffected) {
            affected_.finalize();
            if constexpr (collectTransferStats) stats_counters_.affectedEventsLevel0 += affected_.size();
        }
    }

    /**
     * @brief Collapse (trip, startIndex) entries to one per trip, keeping the MAX startIndex.
     * A trip may be flagged by several triggers at different stops; the largest index is the
     * conservative warm-start boundary (all lower-index decisions are re-evaluated anyway).
     */
    static void aggregateMaxByTrip(std::vector<MinTarget>& targets) {
        std::ranges::sort(targets, [](const MinTarget& a, const MinTarget& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second > b.second;  // largest startIndex first within a trip
        });
        auto out = targets.begin();
        for (auto in = targets.begin(); in != targets.end();) {
            *out = *in;  // first entry per trip already carries the max startIndex
            const TripId trip = in->first;
            do {
                ++in;
            } while (in != targets.end() && in->first == trip);
            ++out;
        }
        targets.erase(out, targets.end());
    }

    /**
     * @brief Full rebuild of minimization flags for ALL trips.
     */
    void buildInitialMinimizedTransfers(const DynamicQueryData& queryData, const int numberOfThreads) {
        queryData_ = &queryData;
        const auto& qd = queryData_->queryData;
        const std::size_t numTrips = qd.routeOfTrip.size();

        // Full rebuild: warm-start boundary = last stop index of each trip, so no stop is skipped.
        std::vector<MinTarget> allTrips(numTrips);
        for (std::size_t i = 0; i < numTrips; ++i) {
            const std::size_t numStops = qd.firstStopEventOfTrip[i + 1] - qd.firstStopEventOfTrip[i];
            allTrips[i] = {TripId(i), StopIndex(numStops - 1)};
        }
        runMinimization(allTrips, numberOfThreads);
    }

    /**
     * @brief Incremental minimization re-run for a set of (flat) trips with warm-start boundaries.
     */
    void updateMinimizedTransfers(std::span<const MinTarget> trips, const DynamicQueryData& queryData,
                                  const int numberOfThreads, const int nowSeconds = noTimeCutoff,
                                  TransferUpdateCounters* counters = nullptr) {
        queryData_ = &queryData;
        runMinimization(trips, numberOfThreads, nowSeconds, counters);
    }

    /**
     * @brief Export the current FULL transfer set in compact layout.
     */
    [[nodiscard]] TripBased::Transfers exportFullTransfers(const DynamicQueryData& queryData,
                                                           const int numberOfThreads,
                                                           PhaseTimings* outPhases = nullptr) const {
        Timer timer;
        TripBased::Transfers result = exporter().exportFull(queryData, numberOfThreads);
        if (outPhases != nullptr) {
            outPhases->exportPhase += std::chrono::microseconds(static_cast<long long>(timer.elapsedMicroseconds()));
        }
        return result;
    }

    /**
     * @brief Export the current REDUCED transfer set in compact layout.
     */
    [[nodiscard]] TripBased::Transfers exportReducedTransfers(const DynamicQueryData& queryData,
                                                              const int numberOfThreads,
                                                              PhaseTimings* outPhases = nullptr) const {
        Timer timer;
        TripBased::Transfers result = exporter().exportReduced(queryData, numberOfThreads);
        if (outPhases != nullptr) {
            outPhases->exportPhase += std::chrono::microseconds(static_cast<long long>(timer.elapsedMicroseconds()));
        }
        return result;
    }

    /**
     * @brief Write TREX customization results back into the persistent store.
     */
    void applyRankRaises(std::span<const RankRaise> raises, const int numberOfThreads = 1) const {
        exporter().applyRankRaises(raises, numberOfThreads);
    }

public:
    // Public: accessed directly by the transfer scenario helpers for store diagnostics.
    Store& store_;

    // Work counters from the most recent applyFullUpdates() call. Only
    // populated in detail builds (collectTransferStats); all-zero otherwise.
    const TransferUpdateCounters& statsCounters() const noexcept { return stats_counters_; }

private:
    [[nodiscard]] inline Exporter exporter() const { return Exporter(store_); }

    /**
     * @brief Merge one thread's affected-set collector. No-op for NullAffectedSink.
     * Call sites are already inside the phase's `omp critical`.
     */
    inline void mergeSink([[maybe_unused]] const AffectedSink& sink) {
        if constexpr (collectsAffected) affected_.merge(sink);
    }

    /**
     * @brief Gather the stop events whose outgoing/incoming transfers must be rediscovered.
     */
    void collectDiscoveryTargets(const DynamicTimeTable::ChangeSummary& changes,
                                 std::vector<PersistentStopEventId>& toDiscoverOutgoing,
                                 std::vector<PersistentStopEventId>& toDiscoverIncoming) const {
        const auto& qd = queryData_->queryData;
        toDiscoverOutgoing.reserve(changes.addedTrips.size() + changes.modifiedEvents.size());
        toDiscoverIncoming.reserve(changes.addedTrips.size() + changes.modifiedEvents.size() +
                                   changes.tripsToRediscoverIncomingDueToCancellation.size());

        for (const auto pTrip : changes.tripsToRediscoverIncomingDueToCancellation) {
            queryData_->appendEventsOfTrip(pTrip, toDiscoverIncoming);
        }
        for (const auto pTrip : changes.addedTrips) {
            queryData_->appendEventsOfTrip(pTrip, toDiscoverOutgoing);
            queryData_->appendEventsOfTrip(pTrip, toDiscoverIncoming);
        }
        {
            const auto keys = std::views::keys(changes.modifiedEvents);
            toDiscoverOutgoing.insert(toDiscoverOutgoing.end(), keys.begin(), keys.end());
            toDiscoverIncoming.insert(toDiscoverIncoming.end(), keys.begin(), keys.end());
        }

        // An earlier departure can newly enable incoming transfers into the same stop on the
        // NEXT trip of the same route: rediscover that event too.
        const std::size_t numTrips = qd.routeOfTrip.size();
        for (const auto& [pEvent, isEarlierDep] : changes.modifiedEvents) {
            if (!isEarlierDep) continue;
            const StopEventId flatEvent = queryData_->persistentToFlatEvent[pEvent];
            const TripId flatTrip = qd.tripOfStopEvent[flatEvent];
            if (static_cast<std::size_t>(flatTrip) + 1 >= numTrips) continue;
            if (qd.routeOfTrip[flatTrip] != qd.routeOfTrip[flatTrip + 1]) continue;
            const auto numStops = qd.firstStopEventOfTrip[flatTrip + 1] - qd.firstStopEventOfTrip[flatTrip];
            toDiscoverIncoming.emplace_back(queryData_->flatToPersistentEvent[flatEvent + numStops]);
        }

        sortUnique(toDiscoverOutgoing);
        sortUnique(toDiscoverIncoming);
    }

    /**
     * @brief Core dispatcher for computing and applying outgoing discovery modifications.
     */
    void updateOutgoingForEvent(const TransferDiscovery& discovery, const Mutator& mutator,
                                PersistentStopEventId event, DiscoveryWorkspace& ws,
                                std::vector<MinTarget>& localTrips, TransferUpdateCounters& lc,
                                AffectedSink& sink) const {
        StopEventId flatFromEvent = queryData_->persistentToFlatEvent[event];
        if (flatFromEvent == noStopEvent) return;

        ws.desired.clear();
        discovery.computeOutgoingTransfers(flatFromEvent, ws, ws.desired);
        if constexpr (collectTransferStats) lc.outgoingEdgesDiscovered += ws.desired.size();

        // Only flag this event's trip for re-minimization if its outgoing set actually
        // changed. If nothing changed here, the trip's minimization is unaffected by this
        // phase (destination-arrival changes are handled separately by Phase 4).
        if (mutator.applyOutgoingDiff(event, ws.desired, lc, sink)) {
            localTrips.emplace_back(queryData_->tripOfEvent(flatFromEvent), queryData_->stopIndexOfEvent(flatFromEvent));
        }
    }

    /**
     * @brief Core dispatcher for computing and applying incoming discovery modifications.
     */
    void updateIncomingForEvent(const TransferDiscovery& discovery, const Mutator& mutator,
                                PersistentStopEventId event, DiscoveryWorkspace& ws,
                                std::vector<MinTarget>& localTrips,
                                std::vector<PendingDominationCleanup>& pendingCleanups,
                                std::vector<PersistentStopEventId>& newlyInserted, TransferUpdateCounters& lc,
                                AffectedSink& sink) const {
        StopEventId flatToEvent = queryData_->persistentToFlatEvent[event];
        if (flatToEvent == noStopEvent) return;

        ws.desired.clear();
        discovery.computeIncomingTransfers(flatToEvent, ws, ws.desired);
        if constexpr (collectTransferStats) lc.incomingEdgesDiscovered += ws.desired.size();
        mutator.applyIncomingDiff(event, flatToEvent, ws.desired, localTrips, pendingCleanups, newlyInserted, lc,
                                  sink);
    }

    /**
     * @brief Shared parallel driver: re-run minimization for the given (flat) trips.
     */
    void runMinimization(std::span<const MinTarget> trips, const int numberOfThreads,
                         const int nowSeconds = noTimeCutoff, TransferUpdateCounters* counters = nullptr) {
        const Minimizer minimizer(store_, *queryData_);
        minimizer.run(
            trips, numberOfThreads, nowSeconds, counters, [] { return AffectedSink{}; },
            [this](const AffectedSink& sink) { mergeSink(sink); });
    }

    // Snapshot of the current active timetable; refreshed at the start of every entry point.
    const DynamicQueryData* queryData_{nullptr};

    // Level-0 affected events of the most recent applyFullUpdates(); always empty for
    // NullAffectedSink.
    AffectedEvents affected_{};

    // Accumulated across the phases of the current applyFullUpdates() call.
    TransferUpdateCounters stats_counters_{};

    // Scratch for applyFullUpdates(), kept alive between calls purely to retain capacity.
    // Cleared at the start of every call; never read across calls.
    std::vector<PersistentStopEventId> toDiscoverOutgoing_{};
    std::vector<PersistentStopEventId> toDiscoverIncoming_{};
    std::vector<MinTarget> globalTripsToMinimize_{};
    std::vector<PendingDominationCleanup> globalPendingCleanups_{};
};

}  // namespace DynamicTB::Preprocessing
