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
 * @brief Sort a vector and drop duplicate entries in place.
 */
template <typename T>
inline void sortUnique(std::vector<T>& v) {
    std::ranges::sort(v);
    v.erase(std::ranges::unique(v).begin(), v.end());
}

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
        // Trips whose minimization must be re-run, kept in flat space (stable for this call).
        std::vector<TripId> globalTripsToMinimize;

        store_.allowTemporaryInconsistent(true);

        // Phase 1: Parallel Cancellations
#pragma omp parallel if (threads > 1)
        {
            std::vector<TripId> localTrips;
            std::vector<NodeID> incomingSnapshot;
#pragma omp for schedule(dynamic, 16)
            for (const auto& cancelledTrip : changes.cancelledTrips) {
                for (const auto event : cancelledTrip.eventsOfCancelledTrips) {
                    store_.copy_incoming(event, incomingSnapshot);
                    for (const auto from : incomingSnapshot) {
                        recordSourceTripOfEvent(from, localTrips);
                    }
                    clearEventTransfers(event);
                }
            }
#pragma omp critical
            globalTripsToMinimize.insert(globalTripsToMinimize.end(), localTrips.begin(), localTrips.end());
        }
        store_.sync_barrier();

        collectDiscoveryTargets(changes, toDiscoverOutgoing, toDiscoverIncoming);

        // Phase 2: Parallel Outgoing Discovery
#pragma omp parallel if (threads > 1)
        {
            std::vector<TripId> localTrips;
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
            std::vector<TripId> localTrips;
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

        // Phase 4: Parallel Processing for Changed Arrivals (later OR earlier).
        // Two effects of a changed arrival on trip T:
        //  (a) SELF: T's own minimization seeds StopLabels from the arrival times of all
        //      of T's stops and accumulates them while scanning stops backward, so a
        //      changed arrival at any stop affects the keep-decisions of EARLIER stops of
        //      T. => T itself must be re-minimized.
        //  (b) UPSTREAM: sources boarding T at stop index j need re-minimization only if
        //      some arrival CHANGED at a stop index > j. maxChangedIndex is the largest
        //      such index, so incoming edges into events at index >= maxChangedIndex are
        //      unaffected and skipped. getEventsOfTrip() returns events in stop-index order.
#pragma omp parallel if (threads > 1)
        {
            std::vector<TripId> localTrips;
#pragma omp for schedule(dynamic, 1024)
            for (const auto& [changedTrip, maxChangedIndex] : changes.tripsWithChangedArrivals) {
                const auto events = queryData_->getEventsOfTrip(changedTrip);
                if (events.empty()) continue;

                // (a) SELF: re-minimize the changed trip itself.
                const StopEventId flatChangedEvent = queryData_->persistentToFlatEvent[events.front()];
                if (flatChangedEvent != noStopEvent) {
                    localTrips.push_back(queryData_->tripOfEvent(flatChangedEvent));
                }

                // (b) UPSTREAM: re-minimize sources feeding stops before the last change.
                const std::size_t limit =
                    std::min(static_cast<std::size_t>(maxChangedIndex), events.size());
                for (std::size_t idx = 0; idx < limit; ++idx) {
                    for (const auto from : store_.incoming_sorted(events[idx])) {
                        recordSourceTripOfEvent(from, localTrips);
                    }
                }
            }
#pragma omp critical
            globalTripsToMinimize.insert(globalTripsToMinimize.end(), localTrips.begin(), localTrips.end());
        }

        sortUnique(globalTripsToMinimize);
        updateMinimizedTransfers(globalTripsToMinimize, queryData, numberOfThreads);
    }

    /**
     * @brief Full rebuild of minimization flags for ALL trips.
     */
    void buildInitialMinimizedTransfers(const DynamicQueryData& queryData, const int numberOfThreads) {
        queryData_ = &queryData;
        const std::size_t numTrips = queryData_->queryData.routeOfTrip.size();

        std::vector<TripId> allTrips(numTrips);
        for (std::size_t i = 0; i < numTrips; ++i) allTrips[i] = TripId(i);
        runMinimization(allTrips, numberOfThreads);
    }

    /**
     * @brief Incremental minimization re-run for a set of (flat) trips.
     */
    void updateMinimizedTransfers(std::span<const TripId> trips, const DynamicQueryData& queryData,
                                  const int numberOfThreads) {
        queryData_ = &queryData;
        runMinimization(trips, numberOfThreads);
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

            const Time fromArrivalTime = queryData.arrivalTimeOfEvent(flatFrom);
            Edge currentEdgeOffset = beginOut[flatFrom];

            for (const auto& [to, meta] : store_.outgoing_sorted(NodeID(pFrom))) {
                if constexpr (OnlyMinimized) {
                    if (!meta.isMinimized) continue;
                }
                const StopEventId flatTo = queryData.persistentToFlatEvent[to];
                if (flatTo == noStopEvent) continue;

                const Edge exportEdge = currentEdgeOffset++;
                const TripId trip = queryData.tripOfEvent(flatTo);
                labels[exportEdge].init(flatTo, trip, qd.firstStopEventOfTrip[trip]);
                travelTime[exportEdge] = static_cast<int>(queryData.departureTimeOfEvent(flatTo) - fromArrivalTime);
            }
        }
        return {std::move(beginOut), std::move(labels), std::move(travelTime)};
    }

    /**
     * @brief Record the (flat) source trip of a stop event for later re-minimization.
     * Converts persistent -> flat exactly once at the store boundary; inactive events are ignored.
     */
    inline void recordSourceTripOfEvent(PersistentStopEventId from, std::vector<TripId>& localTrips) const {
        const StopEventId flatEvent = queryData_->persistentToFlatEvent[from];
        if (flatEvent == noStopEvent) return;
        localTrips.push_back(queryData_->tripOfEvent(flatEvent));
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
            toDiscoverIncoming.append_range(queryData_->getEventsOfTrip(pTrip));
        }
        for (const auto pTrip : changes.addedTrips) {
            toDiscoverOutgoing.append_range(queryData_->getEventsOfTrip(pTrip));
            toDiscoverIncoming.append_range(queryData_->getEventsOfTrip(pTrip));
        }
        toDiscoverOutgoing.append_range(std::views::keys(changes.modifiedEvents));
        toDiscoverIncoming.append_range(std::views::keys(changes.modifiedEvents));

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
    void updateOutgoingForEvent(PersistentStopEventId event, std::vector<TripId>& localTrips) const {
        StopEventId flatFromEvent = queryData_->persistentToFlatEvent[event];
        if (flatFromEvent == noStopEvent) return;

        std::vector<PersistentStopEventId> desired;
        computeOutgoingTransfers(flatFromEvent, desired);

        // Only flag this event's trip for re-minimization if its outgoing set actually
        // changed. If nothing changed here, the trip's minimization is unaffected by this
        // phase (destination-arrival changes are handled separately by Phase 4).
        if (applyOutgoingDiff(event, desired)) {
            localTrips.push_back(queryData_->tripOfEvent(flatFromEvent));
        }
    }

    /**
     * @brief Core dispatcher for computing and applying incoming discovery modifications.
     */
    void updateIncomingForEvent(PersistentStopEventId event, std::vector<TripId>& localTrips,
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
        Time arrTime = queryData_->arrivalTimeOfEvent(flatFromEvent);
        if (arrTime == noTime) return;

        StopId fromStop = queryData_->stopOfEvent(flatFromEvent);
        const auto& qd = queryData_->queryData;
        TripId flatFromTrip = qd.tripOfStopEvent[flatFromEvent];
        RouteId fromRoute = qd.routeOfTrip[flatFromTrip];
        StopIndex fromIndex = queryData_->stopIndexOfEvent(flatFromEvent);

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

                out.push_back(queryData_->persistentEventId(toTrip, segment.stopIndex));
            }
        }
        std::ranges::sort(out);
    }

    /**
     * @brief Compute all feasible incoming transfers to a single stop event.
     */
    void computeIncomingTransfers(StopEventId flatToEvent, std::vector<PersistentStopEventId>& out) const {
        Time toDepTime = queryData_->departureTimeOfEvent(flatToEvent);
        if (toDepTime == noTime) return;

        const auto& qd = queryData_->queryData;
        TripId toTrip = qd.tripOfStopEvent[flatToEvent];
        // Can't transfer to the last stop of a trip
        if (flatToEvent == (qd.firstStopEventOfTrip[toTrip + 1] - 1)) return;

        RouteId toRoute = qd.routeOfTrip[toTrip];
        StopIndex toIndex = queryData_->stopIndexOfEvent(flatToEvent);
        StopId toStop = queryData_->stopOfEvent(flatToEvent);

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

            // Arrival time of a trip at this source's stop index, as int64 for window arithmetic.
            auto arrivalAt = [&](const TripId t) {
                return static_cast<int64_t>(queryData_->arrivalTimeOfEvent(queryData_->stopEventIdOfTripStop(t, src.i)));
            };

            // Leverage Consistency Invariant: if the first trip is noTime, exiting is forbidden for the entire route
            if (Time(qd.eventArrTimes[queryData_->stopEventIdOfTripStop(firstTrip, src.i)]) == noTime) continue;

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
                if (arrivalAt(TripId(firstTrip + mid)) > minArr) {
                    firstIdx = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

            // Iterate forward to collect everything within the valid arrival window
            for (uint32_t idx = static_cast<uint32_t>(firstIdx); idx < numTrips; ++idx) {
                TripId t = TripId(firstTrip + idx);
                if (arrivalAt(t) > maxArr) break;  // Window closed; later trips will arrive too late

                // Filter out U-Turns and same-route forward invalidities
                if (src.route == toRoute && toTrip >= t && toIndex >= src.i) continue;
                if (isUTurn(t, src.i, toTrip, toIndex)) continue;

                PersistentStopEventId pEv = queryData_->flatToPersistentEvent[queryData_->stopEventIdOfTripStop(t, src.i)];
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
     * @brief Two-way sorted set-difference merge of a current edge list against a desired one.
     * `key` projects a current element to its PersistentStopEventId. The callbacks fire on
     * entries present only in `current` (onRemove), only in `desired` (onAdd), and in both
     * (onKeep). Both inputs must be sorted ascending by PersistentStopEventId.
     */
    template <typename CurrentRange, typename Key, typename OnRemove, typename OnAdd, typename OnKeep>
    inline void mergeSortedDiff(CurrentRange&& current, std::span<const PersistentStopEventId> desired, Key key,
                                OnRemove onRemove, OnAdd onAdd, OnKeep onKeep) const {
        auto curr_it = current.begin();
        auto des_it = desired.begin();

        while (curr_it != current.end() && des_it != desired.end()) {
            const PersistentStopEventId currentFrom = key(*curr_it);
            const PersistentStopEventId desiredFrom = *des_it;
            if (currentFrom < desiredFrom) {
                onRemove(*curr_it);
                ++curr_it;
            } else if (currentFrom > desiredFrom) {
                onAdd(desiredFrom);
                ++des_it;
            } else {
                onKeep(*curr_it);
                ++curr_it;
                ++des_it;
            }
        }
        for (; curr_it != current.end(); ++curr_it) onRemove(*curr_it);
        for (; des_it != desired.end(); ++des_it) onAdd(*des_it);
    }

    /**
     * @brief Computes mutations against current outgoing store entries.
     */
    inline bool applyOutgoingDiff(PersistentStopEventId fromEvent,
                                  std::span<const PersistentStopEventId> desired) const {
        auto batch = store_.begin_batch(fromEvent, Store::Direction::Outgoing);
        // Re-minimization of this source is needed only if a candidate that participated
        // in the reduction changed: any ADD (a new candidate can flip other keep-flags),
        // or the removal of a MINIMIZED edge. Removing a non-minimized edge cannot change
        // any keep-flag -- during reduction a non-kept candidate writes no StopLabels, so
        // it has no effect on the decisions of the remaining candidates.
        bool needsRemin = false;
        mergeSortedDiff(
            store_.outgoing_sorted(fromEvent), desired, [](const auto& edge) { return edge.to; },
            [&](const auto& edge) {
                if (edge.meta.isMinimized) needsRemin = true;
                store_.remove_outgoing_edge(batch, edge.to);
            },
            [&](PersistentStopEventId to) {
                store_.add_outgoing_edge(batch, to, TransferMeta{false});
                needsRemin = true;
            },
            [](const auto&) {});
        store_.commit_batch(batch);
        return needsRemin;
    }

    /**
     * @brief Computes mutations against current incoming store entries.
     */
    void applyIncomingDiff(PersistentStopEventId toEvent, StopEventId flatToEvent,
                           std::span<const PersistentStopEventId> desired, std::vector<TripId>& localTrips,
                           std::vector<PendingDominationCleanup>& pendingCleanups) const {
        auto batch = store_.begin_batch(toEvent, Store::Direction::Incoming);

        // New incoming edges whose domination cleanup is deferred.
        // Track reduction targets on edge changes.
        std::vector<PersistentStopEventId> newlyInserted;

        mergeSortedDiff(
            store_.incoming_sorted(toEvent), desired, [](PersistentStopEventId from) { return from; },
            [&](PersistentStopEventId from) {
                store_.remove_incoming_edge(batch, from);
                recordSourceTripOfEvent(from, localTrips);
            },
            [&](PersistentStopEventId from) {
                recordSourceTripOfEvent(from, localTrips);
                store_.add_incoming_edge(batch, from, TransferMeta{false});
                newlyInserted.push_back(from);
            },
            [](PersistentStopEventId) {});

        // Safely commit all incoming operations first, then defer domination cleanup.
        store_.commit_batch(batch);
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
     * @brief If a newly inserted transfer (fromEvent -> flatToEvent) dominates an existing
     * outgoing transfer of fromEvent (same route, same stop index, but a later trip), remove it.
     */
    void dominationCleanupForInsertedTransfer(PersistentStopEventId fromEvent, StopEventId flatToEvent,
                                              std::vector<TripId>& localTrips) const {
        const auto& qd = queryData_->queryData;
        TripId flatToTrip = queryData_->tripOfEvent(flatToEvent);
        RouteId toRoute = qd.routeOfTrip[flatToTrip];
        StopIndex toIndex = queryData_->stopIndexOfEvent(flatToEvent);

        // Snapshot fromEvent's outgoing edges under the store lock. This runs in the
        // parallel incoming phase, where other threads mirror incoming edges into
        // out_[fromEvent] and may reallocate it; a raw span would dangle (heap UAF).
        thread_local std::vector<Store::OutEdge> outgoingSnapshot;
        store_.copy_outgoing(fromEvent, outgoingSnapshot);

        for (const auto& edge : outgoingSnapshot) {
            PersistentStopEventId u2Event = edge.to;
            StopEventId flatU2Event = queryData_->persistentToFlatEvent[u2Event];
            if (flatU2Event == noStopEvent) continue;
            if (queryData_->routeOfEvent(flatU2Event) != toRoute) continue;

            const bool isDominated =
                queryData_->stopIndexOfEvent(flatU2Event) == toIndex && queryData_->tripOfEvent(flatU2Event) > flatToTrip;
            if (!isDominated) continue;

            store_.remove_edge(fromEvent, u2Event);
            if (edge.meta.isMinimized) {
                recordSourceTripOfEvent(fromEvent, localTrips);
            }
            break;
        }
    }

    /**
     * @brief Filtering rule preventing temporal/topological U-turns.
     */
    [[nodiscard]] inline bool isUTurn(const TripId fromTrip, const StopIndex fromIndex, const TripId toTrip,
                                      const StopIndex toIndex) const noexcept {
        return TripBased::isUTurn(fromTrip, fromIndex, toTrip, toIndex, queryData_->queryData);
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
     * @brief Shared parallel driver: re-run minimization for the given (flat) trips.
     * Each thread owns its StopLabel scratch buffers to avoid data races.
     */
    void runMinimization(std::span<const TripId> trips, const int numberOfThreads) const {
        const std::size_t numStops = queryData_->queryData.firstRouteSegmentOfStop.size() - 1;

        const int threads = std::max(1, numberOfThreads);
        omp_set_num_threads(threads);
#pragma omp parallel if (threads > 1)
        {
            std::vector localLabels(numStops, StopLabel());
            std::vector<TransferCandidate> localCandidates;
            int localTimestamp = 0;
#pragma omp for schedule(dynamic, 1)
            for (const TripId trip : trips) {
                reduceTransfersForTrip(trip, localLabels, localCandidates, localTimestamp);
            }
        }
    }

    /**
     * @brief Candidate outgoing transfer during minimization.
     * Holds a pointer directly into the store's outgoing edge so the keep-flag can
     * be written in place (O(1))
     */
    struct TransferCandidate {
        typename Store::OutEdge* edge;
        StopEventId flatTo;
        int destArrivalTime;
    };

    /**
     * @brief Evaluates and assigns minimization metadata for transfers of a given trip.
     */
    void reduceTransfersForTrip(TripId flatTrip, std::vector<StopLabel>& localLabels,
                                std::vector<TransferCandidate>& candidates, int& localTimestamp) const {
        const auto& qd = queryData_->queryData;
        localTimestamp++;

        int numStops = qd.firstStopEventOfTrip[flatTrip + 1] - qd.firstStopEventOfTrip[flatTrip];

        // Scan backward from the destination stop down to the second stop (index 1)
        for (int i = numStops - 1; i > 0; --i) {
            StopEventId flatFromEvent = StopEventId(qd.firstStopEventOfTrip[flatTrip] + i);
            PersistentStopEventId pFromEvent = queryData_->flatToPersistentEvent[flatFromEvent];
            assert(pFromEvent != noPersistentStopEventId);

            int arrivalTime = static_cast<int>(queryData_->arrivalTimeOfEvent(flatFromEvent));
            StopId fromStop = queryData_->stopOfEvent(flatFromEvent);

            // 1. Update labels for the stop itself and its outgoing transfer/footpath neighbors
            localLabels[fromStop].update(localTimestamp, arrivalTime);
            for (const auto edge : qd.transferGraph.edgesFrom(fromStop)) {
                auto toStop = StopId(qd.transferGraph.get(ToVertex, edge));
                const int transferTime = qd.transferGraph.get(TravelTime, edge);
                localLabels[toStop].update(localTimestamp, arrivalTime + transferTime);
            }

            // 2. Gather full unreduced candidates from the edge store.
            candidates.clear();
            for (auto& edge : store_.outgoing_mutable(NodeID(pFromEvent))) {
                StopEventId flatTo = queryData_->persistentToFlatEvent[edge.to];
                if (flatTo != noStopEvent) {
                    candidates.push_back({&edge, flatTo, static_cast<int>(queryData_->arrivalTimeOfEvent(flatTo))});
                }
            }

            // 3. Sort candidates by destination arrival time.
            std::ranges::sort(candidates, [](const TransferCandidate& a, const TransferCandidate& b) {
                if (a.destArrivalTime != b.destArrivalTime) return a.destArrivalTime < b.destArrivalTime;
                return a.edge->to < b.edge->to;
            });

            // 4. Domination profile filtering
            for (const auto& candidate : candidates) {
                bool keep = false;
                TripId toTrip = queryData_->tripOfEvent(candidate.flatTo);
                StopEventId firstEventOfToTrip = qd.firstStopEventOfTrip[toTrip];
                size_t numStopsInToTrip = qd.firstStopEventOfTrip[toTrip + 1] - firstEventOfToTrip;

                for (size_t j =
                         numStopsInToTrip - static_cast<size_t>(StopIndex(candidate.flatTo - firstEventOfToTrip)) - 1;
                     j > 0; --j) {
                    StopEventId destEvent = StopEventId(candidate.flatTo + j);
                    StopId destinationStop = queryData_->stopOfEvent(destEvent);
                    int destinationArrivalTime = static_cast<int>(queryData_->arrivalTimeOfEvent(destEvent));

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
                candidate.edge->meta.isMinimized = keep;
            }
        }
    }

public:
    // Public: accessed directly by the transfer scenario helpers for store diagnostics.
    Store& store_;

private:
    // Snapshot of the current active timetable; refreshed at the start of every entry point.
    const DynamicQueryData* queryData_{nullptr};
};

}  // namespace DynamicTB::Preprocessing
