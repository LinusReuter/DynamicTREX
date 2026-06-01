#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "../../DynamicTimeTable/BuildQueryData.h"
#include "../../../DataStructures/DynamicTimeTable/UpdateTypes.h"
#include "../../../DataStructures/TransferStore/ITransferStore.h"
#include "../../../Helpers/Types.h"

namespace DynamicTB::Preprocessing {

/**
 * Edge metadata stored in the transfer store.
 * Keep this minimal; transfer validity is derived from the timetable.
 */
struct TransferMeta {
    bool isMinimized{false};  // true if kept by minimization
};

/**
 * TransferUpdate
 *
 * Data flow for dynamic transfer updates driven by DynamicTimeTable::ChangeSummary.
 *
 * - Nodes: PersistentStopEventId (0..n-1)
 * - Edges: transfer from source event -> target event
 * - Uses DynamicQueryData for all stop/event/route lookups (no DynamicTimeTable required).
 * - Store contains the FULL set; minimization only toggles TransferMeta::isMinimized.
 * - Domination cleanup is applied only during incoming updates when new transfers are inserted.
 * - No time scaling: uses Time directly.
 */
class TransferUpdate {
public:
    using Store = ITransferStore<PersistentStopEventId, TransferMeta>;
    using DynamicQueryData = DynamicTimeTable::Algo::DynamicQueryData;
    using NodeID = PersistentStopEventId;

    explicit TransferUpdate(Store& store) : store_(store) {}

    /**
     * Full rebuild: clears the store and (re)discovers all outgoing transfers.
     *
     * Implementation notes:
     * - Ensure the store has nodes for all stop events (store.begin_outgoing_init(maxEventId)).
     * - Build outgoing only (no diff); then rebuild/sync incoming once via finish_outgoing_init().
     *
     * This does NOT set minimization flags
     * (call buildInitialMinimizedTransfers).
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

        // 1) Prepare outgoing-only init mode
        const NodeID maxEventId = NodeID(eventCount - 1);
        store_.begin_outgoing_init(maxEventId);

        // 2) Clear store (full clear; incoming will be rebuilt in bulk)
        store_.clear();

        // 3) Discover all outgoing transfers (no diff; init-only add)
        for (std::size_t i = 0; i < eventCount; ++i) {
            PersistentStopEventId event(i);

            // Skip invalid/removed events
            StopEventId flatEvent = queryData_->persistentToFlatEvent[i];
            if (flatEvent == noStopEvent) continue;

            std::vector<PersistentStopEventId> desired;
            computeOutgoingTransfers(event, desired);
            if (!desired.empty()) {
                store_.reserve_outgoing(event, desired.size());
                store_.add_outgoing_edges_init(event, desired, TransferMeta{false});
            }
        }

        // 4) Rebuild incoming / sync_barrier()
        store_.finish_outgoing_init();
    }

    /**
     * Incremental FULL-set update pipeline driven by the latest ChangeSummary:
     * 1) Structural deletions (removed routes + cancelled trips, with redirections)
     * 2) Outgoing discovery for added trips and modified events
     * 3) Incoming discovery for affected targets (added trips + modified events)
     * 4) Domination cleanup triggered from incoming updates when new transfers are inserted
     *
     * Notes:
     * - Ensure the store has nodes for all stop events (store.add_nodes(maxEventId)).
     * - Incoming discovery targets ONLY the specific modified events (including delayed arrivals),
     *   not all events of trips in tripsWithDelayedArrivals; that list is for minimization only.
     * - Transfers that persist must preserve TransferMeta (isMinimized) unchanged.
     *
     * Store interaction uses temporary inconsistency with phase barriers:
     * - Phase 0/1: allowTemporaryInconsistent(true) -> clears/redirections -> sync_barrier()
     * - Phase 2: allowTemporaryInconsistent(true) -> outgoing discovery -> sync_barrier()
     * - Phase 3: allowTemporaryInconsistent(true) -> incoming discovery -> sync_barrier()
     */
    void applyFullUpdates(const DynamicTimeTable::ChangeSummary& changes,
                          const DynamicQueryData& queryData);

    /**
     * Full rebuild of minimization flags for ALL trips.
     * Requires that the FULL set is already present in the store.
     */
    void buildInitialMinimizedTransfers(const DynamicQueryData& queryData);

    /**
     * Incremental minimization re-run for a set of trips.
     * Only toggles TransferMeta::isMinimized in the FULL set.
     *
     * Candidate selection guidance:
     * - Any source trip whose outgoing diff changed (add/remove).
     * - Any source trip that gained/removed incoming transfers due to redirection.
     * - Any source trip whose existing transfers were removed by domination cleanup,
     *   but only if a removed edge had isMinimized=true.
     * - Any source trip that transfers into a trip with delayed arrivals (downstream timing change).
     */
    void updateMinimizedTransfers(const std::vector<PersistentTripId>& trips,
                                  const DynamicQueryData& queryData);

private:
    // === Full-set phase orchestration ===

    /// Clear all transfers for fully removed routes.
    void processRemovedRoutes(const std::vector<PersistentRouteId>& routes);

    /// Handle trip cancellations:
    /// 1) Find next feasible trip on the old route (CancelledTripInfo.oldRouteId).
    /// 2) Redirect incoming transfers to that trip.
    /// 3) Clear outgoing and incoming transfers of the cancelled trip.
    void processCancelledTrips(const std::vector<DynamicTimeTable::CancelledTripInfo>& trips);

    /// Update transfers for newly added or reinserted trips.
    void processAddedTrips(const std::vector<PersistentTripId>& trips);

    /// Recompute transfers for events with in-place time changes.
    void processModifiedEvents(const std::vector<PersistentStopEventId>& events);

    /// For trips with delayed arrivals: flag SOURCE trips of incoming transfers for re-minimization.
    /// This does not perform incoming discovery; only the specific modified events are reprocessed.
    void processDelayedArrivalTrips(const std::vector<PersistentTripId>& trips);



    // === Core transfer update (single event) ===

    /// Update (compute + apply) outgoing transfers for a single stop event.
    /// Requires a valid arrival time (constraints are modeled as invalid times),
    /// and must enforce U-turn filtering + max-wait cap.
    /// Uses an outgoing batch (incoming=false) to apply the diff.
    void updateOutgoingForEvent(PersistentStopEventId event) {
        std::vector<PersistentStopEventId> desired;
        computeOutgoingTransfers(event, desired);
        applyOutgoingDiff(event, desired);
    }

    /// Update (compute + apply) incoming transfers for a single stop event.
    /// Uses store.incoming(target) for the current set and applies diffs via incoming batches.
    /// Requires a valid departure time (constraints are modeled as invalid times),
    /// and must enforce U-turn filtering + max-wait cap.
    /// New edges get isMinimized=false; existing edges preserve metadata.
    /// When new transfers are inserted, trigger domination cleanup on the target line.
    void updateIncomingForEvent(PersistentStopEventId event);

    /// Compute all feasible outgoing transfers from a single stop event.
    /// Skips events with invalid arrival times (constraints modeled as invalid times).
    ///
    /// /* Pseudocode for Outgoing Discovery:
    ///    Let (t, i) be the source trip and stop index.
    ///    Let connectedStops = {q : q is reachable from stop(t,i) via footpath} // O(F)
    ///
    ///    // Optional route-based pruning tracking
    ///    Map<RouteId, Array<TripId>> earliestTrip
    ///
    ///    for each q in connectedStops: // F times
    ///        for each RouteSegment (S, j) at q: // R_avg times
    ///            toTrip = getEarliestTrip(S, j, arrival(t,i) + footpath(stop(t,i)->q)) // O(log T_avg)
    ///
    ///            if toTrip == noTripId: continue // O(1)
    ///            if S == route(t) AND toTrip >= t AND j >= i: continue  // O(1) same-route forward
    ///            if isUTurn(t, i, toTrip, j): continue // O(1)
    ///
    ///            // OPTIONAL route-based pruning:
    ///            // if toTrip >= earliestTrip[S][j]: continue // O(log R_avg)
    ///            // for k = j..lastIndex(S): // O(S_max)
    ///            //     earliestTrip[S][k] = min(earliestTrip[S][k], toTrip)
    ///
    ///            out.push_back(toEvent(toTrip, j)) // O(1) amortized
    ///
    ///    Total Complexity: O(F * R_avg * log T_avg)
    ///    With optional pruning: O(F * R_avg * (log T_avg + log R_avg + S_max))
    /// */
    inline void computeOutgoingTransfers(PersistentStopEventId fromEvent,
                                  std::vector<PersistentStopEventId>& out) const {
        constexpr bool kEnableRouteBasedPruning = true;

        StopEventId flatFromEvent = queryData_->persistentToFlatEvent[fromEvent];
        if (flatFromEvent == noStopEvent) return;

        Time arrTime = arrivalTimeOfEvent(fromEvent);
        if (arrTime == never) return; // invalid arrival time

        StopId fromStop = stopOfEvent(fromEvent);
        TripId flatFromTrip = queryData_->queryData.tripOfStopEvent[flatFromEvent];
        RouteId fromRoute = queryData_->queryData.routeOfTrip[flatFromTrip];
        StopIndex fromIndex = stopIndexOfEvent(fromEvent);

        std::vector<std::pair<StopId, Time>> connectedStops;
        appendConnectedStops(fromStop, connectedStops);

        struct Target {
            RouteId route;
            StopIndex j;
            Time footPathTime;
        };

        std::vector<Target> targets;
        targets.reserve(connectedStops.size() * 4); // heuristic pre-allocation

        for (const auto& [q, footPathTime] : connectedStops) {
            auto routesAtQ = queryData_->queryData.routesContainingStop(q);
            for (const auto& segment : routesAtQ) {
                targets.push_back({segment.routeId, segment.stopIndex, footPathTime});
            }
        }

        if constexpr (kEnableRouteBasedPruning) {
            // Sorting by route, then stop index ascending.
            // By visiting stops sequentially along a route, we can just maintain the
            // earliest trip we can catch so far. Any later stop yielding the same
            // or a later trip is dominated.
            std::sort(targets.begin(), targets.end(), [](const Target& a, const Target& b) {
                if (a.route != b.route) return a.route < b.route;
                if (a.j != b.j) return a.j < b.j;
                return a.footPathTime < b.footPathTime;
            });
        }

        RouteId currentRoute = RouteId(noRouteId);
        TripId minTripSoFar = TripId(noTripId);

        for (const auto& target : targets) {
            if constexpr (kEnableRouteBasedPruning) {
                if (target.route != currentRoute) {
                    currentRoute = target.route;
                    minTripSoFar = TripId(noTripId);
                }
            }

            Time minArr = arrTime + target.footPathTime;
            std::optional<TripId> optToTrip = findEarliestTripOnRoute(target.route, target.j, minArr);
            if (!optToTrip) continue;

            TripId toTrip = *optToTrip;

            if constexpr (kEnableRouteBasedPruning) {
                if (toTrip >= minTripSoFar) continue;
                minTripSoFar = toTrip;
            }

            // same-route forward check
            if (target.route == fromRoute && toTrip >= flatFromTrip && target.j >= fromIndex) continue;


            if (isUTurn(flatFromTrip, fromIndex, toTrip, target.j)) continue;

            PersistentTripId toTripP = queryData_->flatToPersistentTrip[toTrip];
            std::optional<PersistentStopEventId> toEventP = eventId(toTripP, target.j);
            if (toEventP) {
                out.push_back(*toEventP);
            }
        }

        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

    /// Compute all feasible incoming transfers to a single stop event.
    /// Skips events with invalid departure times (constraints modeled as invalid times).
    ///
    /// /* Pseudocode for Incoming Discovery:
    ///    Let (u, j) be the target trip and stop index, and R = route(u).
    ///    Let uPrev = previous trip on R (if any). // O(log T_avg)
    ///    Let connectedStops = {q : q can reach stop(u,j) via footpath} // O(F)
    ///
    ///    for each q in connectedStops: // F times
    ///        maxArr = departure(u, j) - footpath(q->stop(u,j)) // O(1)
    ///        minArr = uPrev ? departure(uPrev, j) - footpath(...) : -infinity // O(1)
    ///
    ///        for each RouteSegment (S, i) at q: // R_avg times
    ///            // This loop is the expensive part
    ///            for each source trip t on S where arrival(t,i) > minArr AND arrival(t,i) <= maxArr: // O(T_s)
    ///
    ///                if S == R AND u >= t AND j >= i: continue  // O(1) same-route forward
    ///                if isUTurn(t, i, u, j): continue // O(1)
    ///
    ///                // Exact mirror of outgoing route-based pruning (OPTIONAL):
    ///                // earliest = findEarliestTripOnRoute(R, j, arrival(t,i) + footpath) // O(log T_avg)
    ///                // if earliest != u: continue
    ///
    ///                out.push_back(fromEvent(t, i)) // O(1) amortized
    ///
    ///    Total Complexity: O(F * R_avg * T_s)
    ///    With optional pruning: O(F * R_avg * T_s * log T_avg)
    ///    (where F=footpath fan-in, R_avg=routes/stop, T_s=source trips in window)
    /// */
    void computeIncomingTransfers(PersistentStopEventId toEvent,
                                  std::vector<PersistentStopEventId>& out) const;

    /// Expand a stop into itself + footpath neighbors with transfer time.
    inline void appendConnectedStops(StopId fromStop,
                              std::vector<std::pair<StopId, Time>>& out) const {
        out.emplace_back(fromStop, 0);

        const auto& tg = queryData_->queryData.transferGraph;
        for (const auto edge : tg.edgesFrom(fromStop)) {
            StopId toStop = StopId(tg.get(ToVertex, edge));
            Time travelTime = Time(tg.get(TravelTime, edge));
            out.emplace_back(toStop, travelTime);
        }
    }

    /// Find earliest feasible event on any route containing the stop.
    std::optional<PersistentStopEventId> findEarliestEvent(StopId stop, Time minDepartureTime,
                                                           PersistentRouteId forbidRoute,
                                                           StopIndex minIndex) const;

    // === Diff / apply ===

    /// Apply the diff between current outgoing edges and the desired set
    /// using an outgoing batch (incoming=false).
    /// Persist TransferMeta for unchanged edges; new edges get isMinimized=false.
    /// Any add/remove marks the source trip for re-minimization.
    inline void applyOutgoingDiff(PersistentStopEventId fromEvent,
                           std::span<const PersistentStopEventId> desired) {
        auto batch = store_.begin_batch(fromEvent, false);

        auto current_span = store_.outgoing(fromEvent);
        std::vector<Store::OutEdge> current(current_span.begin(), current_span.end());

        std::sort(current.begin(), current.end(), [](const auto& a, const auto& b) {
            return a.to < b.to;
        });

        auto curr_it = current.begin();
        auto des_it = desired.begin();

        while (curr_it != current.end() && des_it != desired.end()) {
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

        while (curr_it != current.end()) {
            store_.remove_outgoing_edge(batch, curr_it->to);
            ++curr_it;
        }

        while (des_it != desired.end()) {
            store_.add_outgoing_edge(batch, *des_it, TransferMeta{false});
            ++des_it;
        }

        store_.commit_batch(batch);
    }

    /// Apply the diff between current incoming edges and the desired set
    /// using an incoming batch (incoming=true).
    /// Persist TransferMeta for unchanged edges; new edges get isMinimized=false.
    /// When a new edge is inserted, trigger domination cleanup.
    /// If a removed edge had isMinimized=true, mark the source trip for re-minimization.
    void applyIncomingDiff(PersistentStopEventId toEvent,
                           std::span<const PersistentStopEventId> desired);

    // === Structural removals and redirections ===

    /// Clear all outgoing and incoming transfers for a single trip.
    void clearTripTransfers(PersistentTripId trip);

    /// Clear all transfers for every trip on the route.
    void clearRouteTransfers(PersistentRouteId route);

    /// Redirect all incoming transfers of a cancelled trip to the next feasible trip (same route).
    /// Assumes the cancelled trip's incoming edges are still present when called.
    void redirectIncomingTransfers(const DynamicTimeTable::CancelledTripInfo& trip);

    /// Compute redirection target for a specific transfer source.
    std::optional<PersistentTripId> findNextFeasibleTripOnRoute(PersistentTripId sourceTrip,
                                                                PersistentTripId cancelledTrip,
                                                                PersistentRouteId oldRouteId,
                                                                StopIndex exitIndex,
                                                                StopIndex boardIndex) const;

    // === Domination cleanup (incoming discovery only) ===

    /// Cleanup is applied via the store and reconciled at sync_barrier().
    /// If a removed edge had isMinimized=true, the source trip must be re-minimized.
    ///
    /// For a newly inserted transfer (from -> to), remove transfers from the same source trip
    /// to later trips on the target route/line that are dominated by this transfer.
    ///
    /// /* Pseudocode for Domination Cleanup:
    ///    Let (t, i) be the fromEvent, (u, j) be the toEvent.
    ///
    ///    // O(O_t), where O_t is # of outgoing transfers from (t,i)
    ///    for each transfer (t, i) -> (u2, j2) currently in the store where route(u2) == route(u):
    ///        if u2 > u OR (u2 == u AND j2 > j): // O(1)
    ///            remove (t, i) -> (u2, j2) // O(log O_t) if sorted, O(1) in batch
    ///            if removed transfer had isMinimized=true: // O(1)
    ///                mark trip t for re-minimization
    ///
    ///    Total Complexity: O(O_t)
    /// */
    void dominationCleanupForInsertedTransfer(PersistentStopEventId fromEvent,
                                              PersistentStopEventId toEvent);

    // === Minimization (flag updates only; full set remains intact) ===

    /// Clear all minimization flags for a trip.
    void clearMinimizationFlags(PersistentTripId trip);

    /// Re-run minimization for a trip and set TransferMeta::isMinimized flags.
    void recomputeMinimizedForTrip(PersistentTripId trip);

    /// Re-run minimization for all trips on a route (loop wrapper).
    void recomputeMinimizedForRoute(PersistentRouteId route);

    // === Transfer validity helpers ===

    /// Guard against U-turn transfers that are invalid by topology/time.
    inline bool isUTurn(const TripId fromTrip, const StopIndex fromIndex, const TripId toTrip,
                        const StopIndex toIndex) const noexcept {
        if (fromIndex < 2) return false;
        if (toIndex + 1 >= queryData_->numberOfStopsInTrip(toTrip)) return false;
        if (queryData_->getStop(fromTrip, StopIndex(fromIndex - 1)) != queryData_->getStop(toTrip, StopIndex(toIndex + 1)))
            return false;
        if (queryData_->arrivalTime(fromTrip, StopIndex(fromIndex - 1)) >
            queryData_->departureTime(toTrip, StopIndex(toIndex + 1)))
            return false;
        return true;
    }

    // === Timetable resolution helpers ===
    // #TODO: Later check all calls and optimize variants to minimize persistent to flat conversions in caller and callee

    /// Map (trip, index) -> stop event id (if valid).
    inline std::optional<PersistentStopEventId> eventId(PersistentTripId trip, StopIndex index) const {
        TripId flatTrip = queryData_->persistentToFlatTrip[trip];
        if (flatTrip == noTripId) return std::nullopt;
        StopEventId firstEvent = queryData_->queryData.firstStopEventOfTrip[flatTrip];
        StopEventId flatEvent = StopEventId(firstEvent + index);
        PersistentStopEventId pEvent = queryData_->flatToPersistentEvent[flatEvent];
        return pEvent;
    }

    /// Resolve stop index from a stop event id.
    inline StopIndex stopIndexOfEvent(PersistentStopEventId event) const {
        StopEventId flatEvent = queryData_->persistentToFlatEvent[event];
        TripId flatTrip = queryData_->queryData.tripOfStopEvent[flatEvent];
        StopEventId firstEvent = queryData_->queryData.firstStopEventOfTrip[flatTrip];
        return StopIndex(flatEvent - firstEvent);
    }

    /// Resolve stop id from a stop event id.
    inline StopId stopOfEvent(PersistentStopEventId event) const {
        StopEventId flatEvent = queryData_->persistentToFlatEvent[event];
        return queryData_->queryData.eventLookup[flatEvent].stop;
    }

    /// Access arrival time from a stop event id.
    inline Time arrivalTimeOfEvent(PersistentStopEventId event) const {
        StopEventId flatEvent = queryData_->persistentToFlatEvent[event];
        return Time(queryData_->queryData.eventArrTimes[flatEvent]);
    }

    /// Access departure time from a stop event id.
    inline Time departureTimeOfEvent(PersistentStopEventId event) const {
        StopEventId flatEvent = queryData_->persistentToFlatEvent[event];
        return Time(queryData_->queryData.eventDepTimes[flatEvent]);
    }

    // === Query-data access ===

    /// Earliest feasible trip on a flat route for a given stop index + time.
    /// Must respect a max-wait cap (e.g., 24h) to avoid unrealistic transfers.
    inline std::optional<TripId> findEarliestTripOnRoute(RouteId route, StopIndex stopIndex, Time minDepartureTime) const {
        const auto& routeLabel = queryData_->queryData.routeLabels[route];
        const uint32_t numTrips = routeLabel.numberOfTrips;

        if (numTrips == 0) return std::nullopt;

        const size_t stopSeqStart = queryData_->queryData.firstStopIdOfRoute[route];
        const size_t stopSeqEnd = queryData_->queryData.firstStopIdOfRoute[route + 1];
        const size_t numStops = stopSeqEnd - stopSeqStart;
        if (numStops < 2) return std::nullopt;
        if (static_cast<size_t>(stopIndex) + 1 >= numStops) return std::nullopt; // no departure at last stop

        const size_t baseOffset = static_cast<size_t>(stopIndex) * numTrips;
        if (baseOffset >= routeLabel.departureTimes.size()) return std::nullopt;

        int left = 0;
        int right = static_cast<int>(numTrips) - 1;
        int bestTrip = -1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            Time dep = Time(routeLabel.departureTimes[baseOffset + static_cast<size_t>(mid)]);
            if (dep >= minDepartureTime) {
                bestTrip = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        if (bestTrip != -1) {
            Time dep = Time(routeLabel.departureTimes[baseOffset + static_cast<size_t>(bestTrip)]);
            // 24 hours max-wait cap
            if (dep > minDepartureTime + 24 * 60 * 60) {
                return std::nullopt;
            }
            TripId firstTrip = queryData_->queryData.firstTripOfRoute[route];
            return TripId(firstTrip + bestTrip);
        }
        return std::nullopt;
    }

private:
    const DynamicQueryData* queryData_{nullptr};
    Store& store_;
};

}  // namespace DynamicTB::Preprocessing
