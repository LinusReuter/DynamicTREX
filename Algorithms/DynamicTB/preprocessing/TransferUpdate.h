#pragma once

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "../../../DataStructures/DynamicTimeTable/UpdateTypes.h"
#include "../../../DataStructures/TransferStore/ITransferStore.h"
#include "../../../Helpers/Types.h"
#include "../../DynamicTimeTable/BuildQueryData.h"

namespace DynamicTB::Preprocessing {

/**
 * @brief Edge metadata stored in the transfer store.
 * Keep this minimal; transfer validity is derived from the timetable.
 */
struct TransferMeta {
    bool isMinimized{false};  // true if kept by minimization
};

/**
 * @brief Thread-local label structure matching StopEventGraphBuilder logic.
 */
struct StopLabel {
    int arrivalTime{std::numeric_limits<int>::max()};
    int timestamp{0};

    inline void checkTimestamp(const int newTimestamp) noexcept {
        if (timestamp != newTimestamp) {
            arrivalTime = std::numeric_limits<int>::max();
            timestamp = newTimestamp;
        }
    }

    inline void update(const int newTimestamp, const int newArrivalTime) noexcept {
        checkTimestamp(newTimestamp);
        arrivalTime = std::min(arrivalTime, newArrivalTime);
    }
};

/**
 * @class TransferUpdate
 * @brief Data flow for dynamic transfer updates driven by DynamicTimeTable::ChangeSummary.
 *
 * - Nodes: PersistentStopEventId (0..n-1)
 * - Edges: transfer from source event -> target event
 * - Uses DynamicQueryData for all lookups.
 * - Store contains the FULL set; minimization only toggles TransferMeta::isMinimized.
 */
class TransferUpdate {
public:
    using Store = ITransferStore<PersistentStopEventId, TransferMeta>;
    using DynamicQueryData = DynamicTimeTable::Algo::DynamicQueryData;
    using NodeID = PersistentStopEventId;

    static constexpr bool kEnableSpacialPruning = false;

    /**
     * @brief A domination cleanup deferred out of the parallel incoming phase.
     */
    struct PendingDominationCleanup {
        PersistentStopEventId fromEvent;
        StopEventId flatToEvent;
        
    };

    explicit TransferUpdate(Store& store) : store_(store) {}

    /**
     * @brief Full rebuild: clears the store and (re)discovers all outgoing transfers.
     */
    void buildInitialFullTransfers(const DynamicQueryData& queryData) {
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

        // 3) Discover all outgoing transfers (no diff; init-only add)
        for (std::size_t i = 0; i < eventCount; ++i) {
            PersistentStopEventId event(i);

            // Skip invalid/removed events
            StopEventId flatEvent = queryData_->persistentToFlatEvent[i];
            if (flatEvent == noStopEvent) continue;

            std::vector<PersistentStopEventId> desired;
            computeOutgoingTransfers(flatEvent, desired);
            if (!desired.empty()) {
                store_.reserve_outgoing(event, desired.size());
                store_.add_outgoing_edges_init(event, desired, TransferMeta{false});
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
                          const int numberOfThreads) {
        const int threads = std::max(1, numberOfThreads);
        omp_set_num_threads(threads);
        queryData_ = &queryData;
        const std::size_t eventCount = queryData_->persistentToFlatEvent.size();
        if (eventCount == 0) {
            store_.clear();
            return;
        }

        const NodeID maxEventId = NodeID(eventCount - 1);
        store_.add_nodes(maxEventId);

        std::vector<PersistentStopEventId> toDiscoverOutgoing;
        std::vector<PersistentStopEventId> toDiscoverIncoming;
        std::vector<PersistentTripId> globalTripsToMinimize;

        store_.allowTemporaryInconsistent(true);

        // Phase 1: Parallel Cancellations
#pragma omp parallel if (threads > 1)
        {
            std::vector<PersistentTripId> localTrips;
            std::vector<NodeID> incomingSnapshot;
#pragma omp  for schedule(dynamic, 16)
        for (const auto & cancelledTrip : changes.cancelledTrips) {
            for (const auto event : cancelledTrip.eventsOfCancelledTrips) {
                store_.copy_incoming(event, incomingSnapshot);
                for (auto from : incomingSnapshot) {
                    StopEventId  fEvent = queryData_->persistentToFlatEvent[from];
                    if (fEvent == noStopEvent) continue;
                    TripId sourceTrip = queryData_->queryData.tripOfStopEvent[fEvent];
                    localTrips.push_back(queryData_->flatToPersistentTrip[sourceTrip]);
                }
                clearEventTransfers(event);
            }
        }
#pragma omp critical
            globalTripsToMinimize.insert(globalTripsToMinimize.end(), localTrips.begin(), localTrips.end());
        }
        store_.sync_barrier();

        // Collect discovery targets
        toDiscoverOutgoing.reserve(changes.addedTrips.size() + changes.modifiedEvents.size());
        toDiscoverIncoming.reserve(changes.addedTrips.size() + changes.modifiedEvents.size() +
                                   changes.tripsToRediscoverIncomingDueToCancellation.size());

        for (const auto pTrip : changes.tripsToRediscoverIncomingDueToCancellation) {
            toDiscoverIncoming.append_range(queryData_->getEventsOfTrip(pTrip));
        }
        for (const auto pTrip : changes.addedTrips) {
            toDiscoverOutgoing.append_range(queryData_->getEventsOfTrip(pTrip));
            toDiscoverIncoming.append_range(queryData_->getEventsOfTrip(pTrip));
        }
        toDiscoverOutgoing.append_range(std::views::keys(changes.modifiedEvents));
        toDiscoverIncoming.append_range(std::views::keys(changes.modifiedEvents));

        for (const auto& [pEvent, isEarlierDep] : changes.modifiedEvents) {
            if (isEarlierDep) {
                auto flatEvent = queryData_->persistentToFlatEvent[pEvent];
                auto flatTrip = queryData_->queryData.tripOfStopEvent[flatEvent];
                if (queryData_->queryData.routeOfTrip[flatTrip] == queryData_->queryData.routeOfTrip[flatTrip + 1]) {
                    auto numStops = queryData_->queryData.firstStopEventOfTrip[flatTrip - 1] -
                                    queryData_->queryData.firstStopEventOfTrip[flatTrip];
                    toDiscoverIncoming.emplace_back(queryData_->flatToPersistentEvent[flatEvent + numStops]);
                }
            }
        }

        std::ranges::sort(toDiscoverOutgoing);
        toDiscoverOutgoing.erase(std::ranges::unique(toDiscoverOutgoing).begin(), toDiscoverOutgoing.end());
        std::ranges::sort(toDiscoverIncoming);
        toDiscoverIncoming.erase(std::ranges::unique(toDiscoverIncoming).begin(), toDiscoverIncoming.end());

        // Phase 2: Parallel Outgoing Discovery
#pragma omp parallel if (threads > 1)
        {
            std::vector<PersistentTripId> localTrips;
#pragma omp for schedule(dynamic, 16)
            for (const auto i : toDiscoverOutgoing) {
                updateOutgoingForEvent(i, localTrips);
            }
#pragma omp critical
            globalTripsToMinimize.insert(globalTripsToMinimize.end(), localTrips.begin(), localTrips.end());
        }
        store_.sync_barrier();

        // Phase 3: Parallel Incoming Discovery
        // Domination cleanups are recorded, not executed here: they mutate shared
        // source outgoing lists and must run sequentially after.
        std::vector<PendingDominationCleanup> globalPendingCleanups;
#pragma omp parallel if (threads > 1)
        {
            std::vector<PersistentTripId> localTrips;
            std::vector<PendingDominationCleanup> localCleanups;
#pragma omp for schedule(dynamic, 16)
            for (const auto i : toDiscoverIncoming) {
                updateIncomingForEvent(i, localTrips, localCleanups);
            }
#pragma omp critical
            {
                globalTripsToMinimize.insert(globalTripsToMinimize.end(), localTrips.begin(), localTrips.end());
                globalPendingCleanups.insert(globalPendingCleanups.end(), localCleanups.begin(),
                                             localCleanups.end());
            }
        }
        store_.sync_barrier();

        for (const auto& [fromEvent, flatToEvent] : globalPendingCleanups) {
            dominationCleanupForInsertedTransfer(fromEvent, flatToEvent, globalTripsToMinimize);
        }

        store_.allowTemporaryInconsistent(false);

        // Phase 4: Parallel Processing for Delayed Arrivals
#pragma omp parallel if (threads > 1)
        {
            std::vector<PersistentTripId> localTrips;
#pragma omp for schedule(dynamic, 1024)
            for (const auto tripsWithDelayedArrival : changes.tripsWithDelayedArrivals) {
                for (const auto event : queryData_->getEventsOfTrip(tripsWithDelayedArrival)) {
                    for (auto from : store_.incoming_sorted(event)) {
                        TripId sourceTrip =
                            queryData_->queryData.tripOfStopEvent[queryData_->persistentToFlatEvent[from]];
                        localTrips.push_back(queryData_->flatToPersistentTrip[sourceTrip]);
                    }
                }
            }
#pragma omp critical
            globalTripsToMinimize.insert(globalTripsToMinimize.end(), localTrips.begin(), localTrips.end());
        }

        std::ranges::sort(globalTripsToMinimize);
        auto uniqueRange = std::ranges::unique(globalTripsToMinimize);
        globalTripsToMinimize.erase(uniqueRange.begin(), uniqueRange.end());

        updateMinimizedTransfers(globalTripsToMinimize, queryData, numberOfThreads);
    }

    /**
     * @brief Full rebuild of minimization flags for ALL trips.
     */
    void buildInitialMinimizedTransfers(const DynamicQueryData& queryData, const int numberOfThreads) {
        queryData_ = &queryData;
        const auto& qd = queryData_->queryData;
        const std::size_t numTrips = qd.routeOfTrip.size();
        const std::size_t numStops = qd.firstRouteSegmentOfStop.size() - 1;

        const int threads = std::max(1, numberOfThreads);
        omp_set_num_threads(threads);
#pragma omp parallel if (threads > 1)
        {
            // Thread-local lookup structures to avoid data-races
            std::vector localLabels(numStops, StopLabel());
            int localTimestamp = 0;
#pragma omp for schedule(dynamic, 1)
            for (std::size_t i = 0; i < numTrips; ++i) {
                reduceTransfersForTrip(TripId(i), localLabels, localTimestamp);
            }
        }
    }

    /**
     * @brief Incremental minimization re-run for a set of trips.
     */
    void updateMinimizedTransfers(const std::vector<PersistentTripId>& trips, const DynamicQueryData& queryData,
                                  const int numberOfThreads) {
        queryData_ = &queryData;
        const auto& qd = queryData_->queryData;
        const std::size_t numStops = qd.firstRouteSegmentOfStop.size() - 1;

        const int threads = std::max(1, numberOfThreads);
        omp_set_num_threads(threads);
#pragma omp parallel if (threads > 1)
        {
            // Thread-local lookup structures to avoid data-races
            std::vector localLabels(numStops, StopLabel());
            int localTimestamp = 0;
#pragma omp for schedule(dynamic, 1)
            for (PersistentTripId trip : trips) {
                const auto fTrip = queryData_->persistentToFlatTrip[trip];
                reduceTransfersForTrip(fTrip, localLabels, localTimestamp);
            }
        }
    }

    /**
     * @brief Export the current FULL transfer set in compact layout.
     */
    [[nodiscard]] TripBased::Transfers exportFullTransfers(const DynamicQueryData& queryData,
                                                           const int numberOfThreads) const {
        return exportTransfersImpl<false>(queryData, numberOfThreads);
    }

    /**
     * @brief Export the current REDUCED transfer set in compact layout.
     */
    [[nodiscard]] TripBased::Transfers exportReducedTransfers(const DynamicQueryData& queryData,
                                                              const int numberOfThreads) const {
        return exportTransfersImpl<true>(queryData, numberOfThreads);
    }

private:
    /**
     * @brief Unified multi-pass implementation for exporting graph structures.
     */
    template <bool OnlyMinimized>
    [[nodiscard]] TripBased::Transfers exportTransfersImpl(const DynamicQueryData& queryData,
                                                           const int numberOfThreads) const {
        const auto& qd = queryData.queryData;
        const std::size_t flatEventCount = qd.eventLookup.size();
        const std::size_t persistentCount = queryData.persistentToFlatEvent.size();

        std::vector<Edge> beginOut(flatEventCount + 1, Edge(0));
        const int threads = std::max(1, numberOfThreads);
        omp_set_num_threads(threads);

// Pass 1: Directly map out-degrees to the flat event indices in parallel
#pragma omp parallel for schedule(dynamic, 1024) if (threads > 1)
        for (std::size_t pFrom = 0; pFrom < persistentCount; ++pFrom) {
            const StopEventId flatFrom = queryData.persistentToFlatEvent[pFrom];
            if (flatFrom == noStopEvent) continue;

            Edge degree = Edge(0);
            if constexpr (OnlyMinimized) {
                for (const auto& [to, meta] : store_.outgoing_sorted(NodeID(pFrom))) {
                    if (meta.isMinimized && queryData.persistentToFlatEvent[to] != noStopEvent) {
                        ++degree;
                    }
                }
            } else {
                degree = Edge(store_.out_degree(NodeID(pFrom)));
            }
            beginOut[static_cast<std::size_t>(flatFrom) + 1] = degree;
        }

        // Pass 2: Prefix sum
        for (std::size_t i = 1; i < beginOut.size(); ++i) {
            beginOut[i] = Edge(beginOut[i] + beginOut[i - 1]);
        }

        const std::size_t edgeCount = beginOut.back();
        std::vector<TripBased::EdgeLabel> labels(edgeCount);
        std::vector<int> travelTime(edgeCount);

#pragma omp parallel for schedule(dynamic, 1024) if (threads > 1)
        for (std::size_t pFrom = 0; pFrom < persistentCount; ++pFrom) {
            const StopEventId flatFrom = queryData.persistentToFlatEvent[pFrom];
            if (flatFrom == noStopEvent) continue;

            const Time fromArrivalTime = Time(qd.eventArrTimes[flatFrom]);
            Edge currentEdgeOffset = beginOut[flatFrom];

            for (const auto& [to, meta] : store_.outgoing_sorted(NodeID(pFrom))) {
                if constexpr (OnlyMinimized) {
                    if (!meta.isMinimized) continue;
                }
                const StopEventId flatTo = queryData.persistentToFlatEvent[to];
                if (flatTo == noStopEvent) continue;

                const Edge exportEdge = currentEdgeOffset++;
                const TripId trip = qd.tripOfStopEvent[flatTo];
                labels[exportEdge].init(flatTo, trip, qd.firstStopEventOfTrip[trip]);
                travelTime[exportEdge] = static_cast<int>(Time(qd.eventDepTimes[flatTo]) - fromArrivalTime);
            }
        }
        return {std::move(beginOut), std::move(labels), std::move(travelTime)};
    }

    /**
     * @brief Core dispatcher for computing and applying outgoing discovery modifications.
     */
    void updateOutgoingForEvent(PersistentStopEventId event, std::vector<PersistentTripId>& localTrips) const {
        StopEventId flatFromEvent = queryData_->persistentToFlatEvent[event];
        if (flatFromEvent == noStopEvent) return;

        TripId flatTrip = queryData_->queryData.tripOfStopEvent[flatFromEvent];
        localTrips.push_back(queryData_->flatToPersistentTrip[flatTrip]);

        std::vector<PersistentStopEventId> desired;
        computeOutgoingTransfers(flatFromEvent, desired);
        applyOutgoingDiff(event, desired);
    }

    /**
     * @brief Core dispatcher for computing and applying incoming discovery modifications.
     */
    void updateIncomingForEvent(PersistentStopEventId event, std::vector<PersistentTripId>& localTrips,
                                std::vector<PendingDominationCleanup>& pendingCleanups) const {
        StopEventId flatToEvent = queryData_->persistentToFlatEvent[event];
        if (flatToEvent == noStopEvent) return;

        std::vector<PersistentStopEventId> desired;
        computeIncomingTransfers(flatToEvent, desired);
        applyIncomingDiff(event, flatToEvent, desired, localTrips, pendingCleanups);
    }

    /**
     * @brief Compute all feasible outgoing transfers from a single stop event.
     */
    inline void computeOutgoingTransfers(StopEventId flatFromEvent, std::vector<PersistentStopEventId>& out) const {
        Time arrTime = arrivalTimeOfEvent(flatFromEvent);
        if (arrTime == noTime) return;

        StopId fromStop = stopOfEvent(flatFromEvent);
        const auto& qd = queryData_->queryData;
        TripId flatFromTrip = qd.tripOfStopEvent[flatFromEvent];
        RouteId fromRoute = qd.routeOfTrip[flatFromTrip];
        StopIndex fromIndex = stopIndexOfEvent(flatFromEvent);

        if (fromIndex == StopIndex(0)) return;

        std::vector<std::pair<StopId, Time>> connectedStops;
        appendConnectedStops(fromStop, connectedStops);

        for (const auto& [q, footPathTime] : connectedStops) {
            Time minArr = arrTime + footPathTime;
            for (const auto& segment : qd.routesContainingStop(q)) {
                std::optional<TripId> optToTrip = findEarliestTripOnRoute(segment.routeId, segment.stopIndex, minArr);
                if (!optToTrip) continue;

                TripId toTrip = *optToTrip;
                if (segment.routeId == fromRoute && toTrip >= flatFromTrip && segment.stopIndex >= fromIndex) continue;
                if (isUTurn(flatFromTrip, fromIndex, toTrip, segment.stopIndex)) continue;

                PersistentTripId toTripP = queryData_->flatToPersistentTrip[toTrip];
                if (auto toEventP = eventId(toTripP, segment.stopIndex)) {
                    out.push_back(*toEventP);
                }
            }
        }
        std::ranges::sort(out);
    }

    /**
     * @brief Compute all feasible incoming transfers to a single stop event.
     */
    void computeIncomingTransfers(StopEventId flatToEvent, std::vector<PersistentStopEventId>& out) const {
        Time toDepTime = departureTimeOfEvent(flatToEvent);
        if (toDepTime == noTime) return;

        const auto& qd = queryData_->queryData;
        TripId toTrip = qd.tripOfStopEvent[flatToEvent];
        // Can't transfer to the last stop of a trip
        if (flatToEvent == (qd.firstStopEventOfTrip[toTrip + 1] - 1)) return;

        RouteId toRoute = qd.routeOfTrip[toTrip];
        StopIndex toIndex = stopIndexOfEvent(flatToEvent);
        StopId toStop = stopOfEvent(flatToEvent);

        // Profile optimization: find departure time of the immediate previous trip on the target route.
        // If a source trip can reach the previous trip, it shouldn't transfer to this one.
        TripId firstTripOfToRoute = qd.firstTripOfRoute[toRoute];
        Time prevDepTime = noTime;

        if (toTrip > firstTripOfToRoute) {
            prevDepTime = Time(qd.eventDepTimes[qd.firstStopEventOfTrip[toTrip - 1] + toIndex]);
        }

        std::vector<std::pair<StopId, Time>> connectedStops{{toStop, Time(0)}};
        const auto& rtg = qd.reverseTransferGraph;
        for (const auto edge : rtg.edgesFrom(toStop)) {
            connectedStops.emplace_back(StopId(rtg.get(ToVertex, edge)), Time(rtg.get(TravelTime, edge)));
        }

        // Collect all potential source route segments
        struct SourceSegment {
            RouteId route;
            StopIndex i;
            Time footPathTime;
        };
        std::vector<SourceSegment> sources;
        sources.reserve(connectedStops.size() * 4);  // Heuristic allocation

        for (const auto& [q, footPathTime] : connectedStops) {
            for (const auto& segment : qd.routesContainingStop(q)) {
                if (segment.stopIndex == StopIndex(0)) continue;  // Can't transfer from the first stop of a trip
                sources.push_back({segment.routeId, segment.stopIndex, footPathTime});
            }
        }

        std::ranges::sort(sources, [](const SourceSegment& a, const SourceSegment& b) {
            if (a.route != b.route) return a.route < b.route;
            if (a.i != b.i) return a.i < b.i;
            return a.footPathTime < b.footPathTime;
        });

        // Scan sources and collect valid connections
        for (const auto& src : sources) {
            TripId firstTrip = qd.firstTripOfRoute[src.route];
            uint32_t numTrips = qd.firstTripOfRoute[src.route + 1] - firstTrip;
            if (numTrips == 0) continue;

            // Leverage Consistency Invariant: if the first trip is noTime, exiting is forbidden for the entire route
            StopEventId firstEv = StopEventId(qd.firstStopEventOfTrip[firstTrip] + src.i);
            if (Time(qd.eventArrTimes[firstEv]) == noTime) continue;

            int64_t maxArr = static_cast<int64_t>(toDepTime) - static_cast<int64_t>(src.footPathTime);
            int64_t minArr = (prevDepTime != noTime)
                                 ? static_cast<int64_t>(prevDepTime) - static_cast<int64_t>(src.footPathTime)
                                 : -1;

            // Binary search across trips to locate the first candidate where arrTime > minArr
            int left = 0;
            int right = static_cast<int>(numTrips) - 1;
            int firstIdx = numTrips;

            while (left <= right) {
                int mid = left + (right - left) / 2;
                TripId t = TripId(firstTrip + mid);
                StopEventId ev = StopEventId(qd.firstStopEventOfTrip[t] + src.i);
                int64_t arrTime = static_cast<int64_t>(qd.eventArrTimes[ev]);

                if (arrTime > minArr) {
                    firstIdx = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

            // Iterate forward to collect everything within the valid arrival window
            for (uint32_t idx = static_cast<uint32_t>(firstIdx); idx < numTrips; ++idx) {
                TripId t = TripId(firstTrip + idx);
                StopEventId ev = StopEventId(qd.firstStopEventOfTrip[t] + src.i);
                int64_t arrTime = static_cast<int64_t>(qd.eventArrTimes[ev]);

                if (arrTime > maxArr) break;  // Window closed; later trips will arrive too late

                // Filter out U-Turns and same-route forward invalidities
                if (src.route == toRoute && toTrip >= t && toIndex >= src.i) continue;
                if (isUTurn(t, src.i, toTrip, toIndex)) continue;

                PersistentStopEventId pEv = queryData_->flatToPersistentEvent[ev];
                if (pEv.isValid()) out.push_back(pEv);
            }
        }
        std::ranges::sort(out);
    }

    /**
     * @brief Expand a stop into itself + footpath neighbors with transfer time.
     */
    inline void appendConnectedStops(StopId fromStop, std::vector<std::pair<StopId, Time>>& out) const {
        out.emplace_back(fromStop, 0);
        const auto& tg = queryData_->queryData.transferGraph;
        for (const auto edge : tg.edgesFrom(fromStop)) {
            auto toStop = StopId(tg.get(ToVertex, edge));
            auto travelTime = Time(tg.get(TravelTime, edge));
            out.emplace_back(toStop, travelTime);
        }
    }

    /**
     * @brief Computes mutations against current outgoing store entries.
     */
    inline void applyOutgoingDiff(PersistentStopEventId fromEvent,
                                  std::span<const PersistentStopEventId> desired) const {
        auto batch = store_.begin_batch(fromEvent, Store::Direction::Outgoing);
        auto current_span = store_.outgoing_sorted(fromEvent);

        auto curr_it = current_span.begin();
        auto des_it = desired.begin();

        while (curr_it != current_span.end() && des_it != desired.end()) {
            if (curr_it->to < *des_it) {
                store_.remove_outgoing_edge(batch, curr_it->to);
                ++curr_it;
            } else if (curr_it->to > *des_it) {
                store_.add_outgoing_edge(batch, *des_it, TransferMeta{false});
                ++des_it;
            } else {
                ++curr_it;
                ++des_it;
            }
        }

        while (curr_it != current_span.end()) {
            store_.remove_outgoing_edge(batch, curr_it->to);
            ++curr_it;
        }
        while (des_it != desired.end()) {
            store_.add_outgoing_edge(batch, *des_it, TransferMeta{false});
            ++des_it;
        }

        store_.commit_batch(batch);
    }

    /**
     * @brief Computes mutations against current incoming store entries.
     */
    void applyIncomingDiff(PersistentStopEventId toEvent, StopEventId flatToEvent,
                           std::span<const PersistentStopEventId> desired, std::vector<PersistentTripId>& localTrips,
                           std::vector<PendingDominationCleanup>& pendingCleanups) const {
        auto batch = store_.begin_batch(toEvent, Store::Direction::Incoming);
        auto current_span = store_.incoming_sorted(toEvent);
        auto curr_it = current_span.begin();
        auto des_it = desired.begin();

        // 1. Create a buffer to track new edges requiring cleanup
        std::vector<PersistentStopEventId> newlyInserted;

        while (curr_it != current_span.end() && des_it != desired.end()) {
            PersistentStopEventId currentFrom = *curr_it;
            PersistentStopEventId desiredFrom = *des_it;

            if (currentFrom < desiredFrom) {
                store_.remove_incoming_edge(batch, currentFrom);
                TripId flatTrip = queryData_->queryData.tripOfStopEvent[queryData_->persistentToFlatEvent[currentFrom]];
                localTrips.push_back(queryData_->flatToPersistentTrip[flatTrip]);
                ++curr_it;
            } else if (currentFrom > desiredFrom) {
                TripId flatTrip = queryData_->queryData.tripOfStopEvent[queryData_->persistentToFlatEvent[currentFrom]];
                localTrips.push_back(queryData_->flatToPersistentTrip[flatTrip]);
                store_.add_incoming_edge(batch, desiredFrom, TransferMeta{false});

                // 2. Track the insertion instead of calling dominationCleanup immediately
                newlyInserted.push_back(desiredFrom);
                ++des_it;
            } else {
                TripId flatTrip = queryData_->queryData.tripOfStopEvent[queryData_->persistentToFlatEvent[currentFrom]];
                localTrips.push_back(queryData_->flatToPersistentTrip[flatTrip]);
                ++curr_it;
                ++des_it;
            }
        }

        while (curr_it != current_span.end()) {
            store_.remove_incoming_edge(batch, *curr_it);
            TripId flatTrip = queryData_->queryData.tripOfStopEvent[queryData_->persistentToFlatEvent[*curr_it]];
            localTrips.push_back(queryData_->flatToPersistentTrip[flatTrip]);
            ++curr_it;
        }
        while (des_it != desired.end()) {
            TripId flatTrip = queryData_->queryData.tripOfStopEvent[queryData_->persistentToFlatEvent[*des_it]];
            localTrips.push_back(queryData_->flatToPersistentTrip[flatTrip]);
            store_.add_incoming_edge(batch, *des_it, TransferMeta{false});
            newlyInserted.push_back(*des_it);
            ++des_it;
        }

        // 3. Safely commit all incoming operations first
        store_.commit_batch(batch);

        // 4. Defer domination cleanup:
        for (const auto& fromEvent : newlyInserted) {
            pendingCleanups.push_back({fromEvent, flatToEvent});
        }
    }

    /**
     * @brief Clear all transfers for a single stop event (incoming + outgoing).
     */
    inline void clearEventTransfers(PersistentStopEventId event) const {
        store_.clear_outgoing(event);
        store_.clear_incoming(event);
    }

    // === Domination cleanup (incoming discovery only) ===

    /**
     * Domination Cleanup
     * Checks of a given Transfer dominates other transfers in the store.
     * @param fromEvent
     * @param toEvent
     * @param flatToEvent
     * @param localTrips
     */
    void dominationCleanupForInsertedTransfer(PersistentStopEventId fromEvent,
                                              StopEventId flatToEvent, std::vector<PersistentTripId>& localTrips) const {
        const auto& qd = queryData_->queryData;
        TripId flatToTrip = qd.tripOfStopEvent[flatToEvent];
        RouteId toRoute = qd.routeOfTrip[flatToTrip];
        StopIndex toIndex = stopIndexOfEvent(flatToEvent);

        // Snapshot fromEvent's outgoing edges under the store lock. This runs in the
        // parallel incoming phase, where other threads mirror incoming edges into
        // out_[fromEvent] and may reallocate it; a raw span would dangle (heap UAF).
        thread_local std::vector<Store::OutEdge> outgoingSnapshot;
        store_.copy_outgoing(fromEvent, outgoingSnapshot);

        for (const auto& edge : outgoingSnapshot) {
            PersistentStopEventId u2Event = edge.to;
            StopEventId flatU2Event = queryData_->persistentToFlatEvent[u2Event];
            if (flatU2Event == noStopEvent) continue;

            if (qd.routeOfTrip[qd.tripOfStopEvent[flatU2Event]] == toRoute) {
                StopIndex u2Index = stopIndexOfEvent(flatU2Event);
                bool isDominated = false;

                if (u2Index == toIndex && qd.tripOfStopEvent[flatU2Event] > flatToTrip) {
                    isDominated = true;
                }

                if (isDominated) {
                    store_.remove_edge(fromEvent, u2Event);
                    if (edge.meta.isMinimized) {
                        TripId flatTrip = qd.tripOfStopEvent[queryData_->persistentToFlatEvent[fromEvent]];
                        localTrips.push_back(queryData_->flatToPersistentTrip[flatTrip]);
                    }
                    break;
                }
            }
        }
    }

    /**
     * @brief Filtering rule preventing temporal/topological U-turns.
     */
    [[nodiscard]] inline bool isUTurn(const TripId fromTrip, const StopIndex fromIndex, const TripId toTrip,
                                      const StopIndex toIndex) const noexcept {
        if (fromIndex < 2) return false;
        if (toIndex + 1 >= queryData_->numberOfStopsInTrip(toTrip)) return false;
        if (queryData_->getStop(fromTrip, StopIndex(fromIndex - 1)) !=
            queryData_->getStop(toTrip, StopIndex(toIndex + 1)))
            return false;
        if (queryData_->arrivalTime(fromTrip, StopIndex(fromIndex - 1)) >
            queryData_->departureTime(toTrip, StopIndex(toIndex + 1)))
            return false;
        return true;
    }

    // === Timetable resolution helpers ===
    // #TODO: Later check all calls and optimize variants to minimize persistent to flat conversions in caller and
    // callee

    /**
     * @brief Map (trip, index) -> stop event id (if valid).
     */
    [[nodiscard]] inline std::optional<PersistentStopEventId> eventId(PersistentTripId trip, StopIndex index) const {
        TripId flatTrip = queryData_->persistentToFlatTrip[trip];
        if (flatTrip == noTripId) return std::nullopt;
        return queryData_
            ->flatToPersistentEvent[StopEventId(queryData_->queryData.firstStopEventOfTrip[flatTrip] + index)];
    }

    /**
     * @brief Resolves StopIndex directly from a flat stop event ID.
     */
    [[nodiscard]] inline StopIndex stopIndexOfEvent(StopEventId flatEvent) const noexcept {
        const auto& qd = queryData_->queryData;
        return StopIndex(flatEvent - qd.firstStopEventOfTrip[qd.tripOfStopEvent[flatEvent]]);
    }

    /**
     * @brief Resolves StopId directly from a flat stop event ID.
     */
    [[nodiscard]] inline StopId stopOfEvent(StopEventId flatEvent) const noexcept {
        return queryData_->queryData.eventLookup[flatEvent].stop;
    }

    /**
     * @brief Accesses arrival Time directly from a flat stop event ID.
     */
    [[nodiscard]] inline Time arrivalTimeOfEvent(StopEventId flatEvent) const noexcept {
        return Time(queryData_->queryData.eventArrTimes[flatEvent]);
    }

    /**
     * @brief Accesses departure Time directly from a flat stop event ID.
     */
    [[nodiscard]] inline Time departureTimeOfEvent(StopEventId flatEvent) const noexcept {
        return Time(queryData_->queryData.eventDepTimes[flatEvent]);
    }

    /**
     * @brief Finds the earliest available trip matching arrival constraints via binary search.
     */
    [[nodiscard]] inline std::optional<TripId> findEarliestTripOnRoute(const RouteId route, const StopIndex stopIndex,
                                                                       const Time minDepartureTime) const {
        const auto& routeLabel = queryData_->queryData.routeLabels[route];
        const uint32_t numTrips = routeLabel.numberOfTrips;

        if (numTrips == 0) return std::nullopt;

        const size_t stopSeqStart = queryData_->queryData.firstStopIdOfRoute[route];
        const size_t stopSeqEnd = queryData_->queryData.firstStopIdOfRoute[route + 1];
        const size_t numStops = stopSeqEnd - stopSeqStart;
        if (numStops < 2) return std::nullopt;
        if (static_cast<size_t>(stopIndex) + 1 >= numStops) return std::nullopt;  // no departure at last stop

        const size_t baseOffset = static_cast<size_t>(stopIndex) * numTrips;
        if (baseOffset >= routeLabel.departureTimes.size()) return std::nullopt;

        int left = 0;
        int right = static_cast<int>(numTrips) - 1;
        int bestTrip = -1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (Time(routeLabel.departureTimes[baseOffset + static_cast<size_t>(mid)]) >= minDepartureTime) {
                bestTrip = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        if (bestTrip != -1) {
            return TripId(queryData_->queryData.firstTripOfRoute[route] + bestTrip);
        }
        return std::nullopt;
    }

    /**
     * @brief Evaluates and assigns minimization metadata for transfers of a given trip.
     */
    void reduceTransfersForTrip(TripId flatTrip, std::vector<StopLabel>& localLabels, int& localTimestamp) const {
        const auto& qd = queryData_->queryData;
        localTimestamp++;

        int numStops = qd.firstStopEventOfTrip[flatTrip + 1] - qd.firstStopEventOfTrip[flatTrip];

        // Scan backward from the destination stop down to the second stop (index 1)
        for (int i = numStops - 1; i > 0; --i) {
            StopEventId flatFromEvent = StopEventId(qd.firstStopEventOfTrip[flatTrip] + i);
            PersistentStopEventId pFromEvent = queryData_->flatToPersistentEvent[flatFromEvent];
            assert(pFromEvent != noPersistentStopEventId);

            int arrivalTime = static_cast<int>(qd.eventArrTimes[flatFromEvent]);
            StopId fromStop = qd.eventLookup[flatFromEvent].stop;

            // 1. Update labels for the stop itself and its outgoing transfer/footpath neighbors
            localLabels[fromStop].update(localTimestamp, arrivalTime);
            for (const auto edge : qd.transferGraph.edgesFrom(fromStop)) {
                auto toStop = StopId(qd.transferGraph.get(ToVertex, edge));
                const int transferTime = qd.transferGraph.get(TravelTime, edge);
                localLabels[toStop].update(localTimestamp, arrivalTime + transferTime);
            }

            // 2. Gather full unreduced candidates from the edge store
            struct TransferCandidate {
                PersistentStopEventId to;
                StopEventId flatTo;
                int destArrivalTime;
            };

            std::vector<TransferCandidate> candidates;
            for (const auto& [to, meta] : store_.outgoing_sorted(NodeID(pFromEvent))) {
                StopEventId flatTo = queryData_->persistentToFlatEvent[to];
                if (flatTo != noStopEvent) {
                    candidates.push_back({to, flatTo, static_cast<int>(qd.eventArrTimes[flatTo])});
                }
            }

            // 3. Sort candidates by destination arrival time matching the original reduction requirements
            std::ranges::stable_sort(candidates, [](const TransferCandidate& a, const TransferCandidate& b) {
                return a.destArrivalTime < b.destArrivalTime;
            });

            // 4. Domination profile filtering
            std::vector<PersistentStopEventId> keepTransfers;
            for (const auto& candidate : candidates) {
                bool keep = false;
                TripId toTrip = qd.tripOfStopEvent[candidate.flatTo];
                StopEventId firstEventOfToTrip = qd.firstStopEventOfTrip[toTrip];
                size_t numStopsInToTrip = qd.firstStopEventOfTrip[toTrip + 1] - firstEventOfToTrip;

                for (size_t j =
                         numStopsInToTrip - static_cast<size_t>(StopIndex(candidate.flatTo - firstEventOfToTrip)) - 1;
                     j > 0; --j) {
                    StopEventId destEvent = StopEventId(candidate.flatTo + j);
                    StopId destinationStop = qd.eventLookup[destEvent].stop;
                    int destinationArrivalTime = static_cast<int>(qd.eventArrTimes[destEvent]);

                    localLabels[destinationStop].checkTimestamp(localTimestamp);
                    if (localLabels[destinationStop].arrivalTime > destinationArrivalTime) {
                        localLabels[destinationStop].arrivalTime = destinationArrivalTime;
                        keep = true;
                    }

                    for (const auto edge : qd.transferGraph.edgesFrom(destinationStop)) {
                        StopId arrivalStop = StopId(qd.transferGraph.get(ToVertex, edge));
                        int arrivalTimeAtStop = destinationArrivalTime + qd.transferGraph.get(TravelTime, edge);

                        localLabels[arrivalStop].checkTimestamp(localTimestamp);
                        if (localLabels[arrivalStop].arrivalTime > arrivalTimeAtStop) {
                            localLabels[arrivalStop].arrivalTime = arrivalTimeAtStop;
                            keep = true;
                        }
                    }
                }
                store_.update_edge_meta(pFromEvent, candidate.to, TransferMeta{keep});
            }
        }
    }

public:
    const DynamicQueryData* queryData_{nullptr};
    Store& store_;
};

}  // namespace DynamicTB::Preprocessing
