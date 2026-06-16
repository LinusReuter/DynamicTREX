#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "../../../DataStructures/DynamicTimeTable/UpdateTypes.h"
#include "../../../DataStructures/TransferStore/ITransferStore.h"
#include "../../../Helpers/Types.h"
#include "../../DynamicTimeTable/BuildQueryData.h"

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

    static constexpr bool kEnableSpacialPruning = false;

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
     * 1) Structural deletions (cancelled trips, with optional redirections)
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
     * - Phase 0/1: allowTemporaryInconsistent(true) -> redirections + clears -> sync_barrier()
     * - Phase 2: allowTemporaryInconsistent(true) -> outgoing discovery -> sync_barrier()
     * - Phase 3: allowTemporaryInconsistent(true) -> incoming discovery -> sync_barrier()
     */
    void applyFullUpdates(const DynamicTimeTable::ChangeSummary& changes, const DynamicQueryData& queryData) {
        queryData_ = &queryData;

        const std::size_t eventCount = queryData_->persistentToFlatEvent.size();
        if (eventCount == 0) {
            store_.clear();
            return;
        }

        const NodeID maxEventId = NodeID(eventCount - 1);
        store_.add_nodes(maxEventId);

        store_.allowTemporaryInconsistent(true);
        processCancelledTrips(changes.cancelledTrips);
        store_.sync_barrier();

        // Outgoing Phase
        processAddedTripsOutgoing(changes.addedTrips);
        processEventsOutgoing(changes.modifiedEvents);
        store_.sync_barrier();

        // Incoming Phase
        // #TODO: When updates cause earlier departure incoming discovey on pedecessor event needed?
        processAddedTripsIncoming(changes.addedTrips);
        processEventsIncoming(changes.modifiedEvents);
        store_.sync_barrier();
        store_.allowTemporaryInconsistent(false);

        // TODO minimization
    }

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
    void updateMinimizedTransfers(const std::vector<PersistentTripId>& trips, const DynamicQueryData& queryData);

    // === Transfer Set Export ===

    /// Export the current FULL transfer set in the compact TripBased query layout.
    ///
    /// The exported graph uses flat stop-event ids as vertices, matching
    /// TripBased::Transfers. Persistent ids from the dynamic store are translated
    /// through DynamicQueryData. Invalid/removed persistent events are skipped.
    [[nodiscard]] TripBased::Transfers exportFullTransfers(const DynamicQueryData& queryData) const {
        const auto& qd = queryData.queryData;
        const std::size_t flatEventCount = qd.eventLookup.size();

        std::vector<Edge> beginOut(flatEventCount + 1, Edge(0));

        for (std::size_t pFrom = 0; pFrom < queryData.persistentToFlatEvent.size(); ++pFrom) {
            const StopEventId flatFrom = queryData.persistentToFlatEvent[pFrom];
            if (flatFrom == noStopEvent) continue;

            std::size_t validOutgoing = 0;
            for (const auto& [to, meta] : store_.outgoing_sorted(NodeID(pFrom))) {
                const NodeID pTo = to;
                if (static_cast<std::size_t>(pTo) >= queryData.persistentToFlatEvent.size()) continue;

                if (const StopEventId flatTo = queryData.persistentToFlatEvent[pTo]; flatTo == noStopEvent) continue;

                ++validOutgoing;
            }

            beginOut[static_cast<std::size_t>(flatFrom) + 1] = Edge(validOutgoing);
        }

        for (std::size_t i = 1; i < beginOut.size(); ++i) {
            beginOut[i] = Edge(beginOut[i] + beginOut[i - 1]);
        }

        const std::size_t edgeCount = beginOut.back();
        std::vector<TripBased::EdgeLabel> labels(edgeCount);
        std::vector<int> travelTime(edgeCount);

        std::vector<Edge> nextEdge = beginOut;

        for (std::size_t pFrom = 0; pFrom < queryData.persistentToFlatEvent.size(); ++pFrom) {
            const StopEventId flatFrom = queryData.persistentToFlatEvent[pFrom];
            if (flatFrom == noStopEvent) continue;

            const Time fromArrivalTime = Time(qd.eventArrTimes[flatFrom]);

            for (const auto& [to, meta] : store_.outgoing_sorted(NodeID(pFrom))) {
                const NodeID pTo = to;
                if (static_cast<std::size_t>(pTo) >= queryData.persistentToFlatEvent.size()) continue;

                const StopEventId flatTo = queryData.persistentToFlatEvent[pTo];
                if (flatTo == noStopEvent) continue;

                const Edge exportEdge = nextEdge[flatFrom]++;
                const TripId trip = qd.tripOfStopEvent[flatTo];
                const StopEventId firstEvent = qd.firstStopEventOfTrip[trip];

                labels[exportEdge].init(flatTo, trip, firstEvent);
                travelTime[exportEdge] = static_cast<int>(Time(qd.eventDepTimes[flatTo]) - fromArrivalTime);
            }
        }

        return {std::move(beginOut), std::move(labels), std::move(travelTime)};
    }

    // Export Minimized Transfers

private:
    // === Full-set phase orchestration ===

    /// Handle trip cancellations:
    /// 1) Redirect incoming transfers to the next active trip (if any).
    /// 2) Clear outgoing and incoming transfers of the cancelled trip.
    void processCancelledTrips(const std::vector<DynamicTimeTable::CancelledTripInfo>& trips) const {
        for (const auto& trip : trips) {
            if (trip.eventsOfCancelledTrips.empty()) continue;
            redirectIncomingTransfers(trip);
            clearTripTransfers(trip.eventsOfCancelledTrips);
        }
    }

    /// Update outgoing transfers for newly added or reinserted trips.
    void processAddedTripsOutgoing(const std::vector<PersistentTripId>& trips) const {
        for (const auto pTrip : trips) {
            processEventsOutgoing(queryData_->getEventsOfTrip(pTrip));
        }
    }

    /// Update incoming transfers for newly added or reinserted trips.
    void processAddedTripsIncoming(const std::vector<PersistentTripId>& trips) const {
        for (const auto pTrip : trips) {
            processEventsIncoming(queryData_->getEventsOfTrip(pTrip));
        }
    }

    void processEventsOutgoing(const std::vector<PersistentStopEventId>& events) const {
        for (const auto event : events) {
            updateOutgoingForEvent(event);
        }
    }
    void processEventsIncoming(const std::vector<PersistentStopEventId>& events) const {
        for (const auto event : events) {
            updateIncomingForEvent(event);
        }
    }

    /// For trips with delayed arrivals: flag SOURCE trips of incoming transfers for re-minimization.
    /// This does not perform incoming discovery; only the specific modified events are reprocessed.
    void processDelayedArrivalTrips(const std::vector<PersistentTripId>& trips);

    // === Core transfer update (single event) ===

    /// Update (compute + apply) outgoing transfers for a single stop event.
    /// Requires a valid arrival time (constraints are modeled as invalid times),
    /// and must enforce U-turn filtering + max-wait cap.
    /// Uses an outgoing batch (Direction::Outgoing) to apply the diff.
    void updateOutgoingForEvent(PersistentStopEventId event) const {
        std::vector<PersistentStopEventId> desired;
        computeOutgoingTransfers(event, desired);
        applyOutgoingDiff(event, desired);
    }

    /// Update (compute + apply) incoming transfers for a single stop event.
    /// Uses store.incoming_sorted(target) for the current set and applies diffs via incoming batches.
    /// Requires a valid departure time (constraints are modeled as invalid times),
    /// and must enforce U-turn filtering + max-wait cap.
    /// New edges get isMinimized=false; existing edges preserve metadata.
    /// When new transfers are inserted, trigger domination cleanup on the target line.
    void updateIncomingForEvent(PersistentStopEventId event) const {
        std::vector<PersistentStopEventId> desired;
        computeIncomingTransfers(event, desired);
        applyIncomingDiff(event, desired);
    }

    /// Compute all feasible outgoing transfers from a single stop event.
    inline void computeOutgoingTransfers(PersistentStopEventId fromEvent,
                                         std::vector<PersistentStopEventId>& out) const {
        StopEventId flatFromEvent = queryData_->persistentToFlatEvent[fromEvent];
        if (flatFromEvent == noStopEvent) return;

        Time arrTime = arrivalTimeOfEvent(fromEvent);
        if (arrTime == noTime) return;  // invalid arrival time

        StopId fromStop = stopOfEvent(fromEvent);
        TripId flatFromTrip = queryData_->queryData.tripOfStopEvent[flatFromEvent];
        RouteId fromRoute = queryData_->queryData.routeOfTrip[flatFromTrip];
        StopIndex fromIndex = stopIndexOfEvent(fromEvent);

        if (fromIndex == StopIndex(0)) return;

        std::vector<std::pair<StopId, Time>> connectedStops;
        appendConnectedStops(fromStop, connectedStops);

        if constexpr (kEnableSpacialPruning) {
            // Spatial pruning requires tracking routes, meaning we must gather and sort.
            struct Target {
                RouteId route;
                StopIndex j;
                Time footPathTime;
            };

            std::vector<Target> targets;
            targets.reserve(connectedStops.size() * 4);  // heuristic pre-allocation

            for (const auto& [q, footPathTime] : connectedStops) {
                for (const auto& segment : queryData_->queryData.routesContainingStop(q)) {
                    targets.push_back({segment.routeId, segment.stopIndex, footPathTime});
                }
            }

            // Sort by route, then stop index ascending for sequential spatial dominance evaluation
            std::ranges::sort(targets.begin(), targets.end(), [](const Target& a, const Target& b) {
                if (a.route != b.route) return a.route < b.route;
                if (a.j != b.j) return a.j < b.j;
                return a.footPathTime < b.footPathTime;
            });

            auto currentRoute = RouteId(noRouteId);
            auto minTripSoFar = TripId(noTripId);

            for (const auto& target : targets) {
                if (target.route != currentRoute) {
                    currentRoute = target.route;
                    minTripSoFar = TripId(noTripId);
                }

                Time minArr = arrTime + target.footPathTime;
                std::optional<TripId> optToTrip = findEarliestTripOnRoute(target.route, target.j, minArr);
                if (!optToTrip) continue;

                TripId toTrip = *optToTrip;

                // Spatial Domination: Ignore if we could catch an earlier (or same) trip at a prior stop
                if (toTrip >= minTripSoFar) continue;
                minTripSoFar = toTrip;

                // Same-route forward check & U-Turn prevention
                if (target.route == fromRoute && toTrip >= flatFromTrip && target.j >= fromIndex) continue;
                if (isUTurn(flatFromTrip, fromIndex, toTrip, target.j)) continue;

                PersistentTripId toTripP = queryData_->flatToPersistentTrip[toTrip];
                if (auto toEventP = eventId(toTripP, target.j)) {
                    out.push_back(*toEventP);
                }
            }
        } else {
            // No spatial pruning.
            for (const auto& [q, footPathTime] : connectedStops) {
                Time minArr = arrTime + footPathTime;
                for (const auto& segment : queryData_->queryData.routesContainingStop(q)) {
                    std::optional<TripId> optToTrip =
                        findEarliestTripOnRoute(segment.routeId, segment.stopIndex, minArr);
                    if (!optToTrip) continue;

                    TripId toTrip = *optToTrip;

                    if (segment.routeId == fromRoute && toTrip >= flatFromTrip && segment.stopIndex >= fromIndex)
                        continue;
                    if (isUTurn(flatFromTrip, fromIndex, toTrip, segment.stopIndex)) continue;

                    PersistentTripId toTripP = queryData_->flatToPersistentTrip[toTrip];
                    if (auto toEventP = eventId(toTripP, segment.stopIndex)) {
                        out.push_back(*toEventP);
                    }
                }
            }
        }

        std::ranges::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

    /// Compute all feasible incoming transfers to a single stop event.
    void computeIncomingTransfers(PersistentStopEventId toEvent, std::vector<PersistentStopEventId>& out) const {
        StopEventId flatToEvent = queryData_->persistentToFlatEvent[toEvent];
        if (flatToEvent == noStopEvent) return;

        Time toDepTime = departureTimeOfEvent(toEvent);
        if (toDepTime == noTime) return;  // Boarding not allowed at this target event

        const auto& qd = queryData_->queryData;
        TripId toTrip = qd.tripOfStopEvent[flatToEvent];
        // Can't transfer to the last stop of a trip
        if (flatToEvent == (qd.firstStopEventOfTrip[toTrip + 1] -1)) return;
        RouteId toRoute = qd.routeOfTrip[toTrip];
        StopIndex toIndex = stopIndexOfEvent(toEvent);
        StopId toStop = stopOfEvent(toEvent);

        // Profile optimization: find departure time of the immediate previous trip on the target route.
        // If a source trip can reach the previous trip, it shouldn't transfer to this one.
        TripId firstTripOfToRoute = qd.firstTripOfRoute[toRoute];
        Time prevDepTime = noTime;

        if (toTrip > firstTripOfToRoute) {
            TripId uPrev = TripId(toTrip - 1);
            StopEventId prevFlatEvent = StopEventId(qd.firstStopEventOfTrip[uPrev] + toIndex);
            prevDepTime = Time(qd.eventDepTimes[prevFlatEvent]);
        }

        // Gather all incoming connected stops (self-transfer + reverse footpaths)
        std::vector<std::pair<StopId, Time>> connectedStops;
        connectedStops.emplace_back(toStop, Time(0));

        const auto& rtg = qd.reverseTransferGraph;
        for (const auto edge : rtg.edgesFrom(toStop)) {
            StopId q = StopId(rtg.get(ToVertex, edge));
            Time travelTime = Time(rtg.get(TravelTime, edge));
            connectedStops.emplace_back(q, travelTime);
        }

        // Collect all potential source route segments
        struct SourceSegment {
            RouteId route;
            StopIndex i;
            Time footPathTime;
        };
        std::vector<SourceSegment> sources;
        sources.reserve(connectedStops.size() * 4); // Heuristic allocation

        for (const auto& [q, footPathTime] : connectedStops) {
            for (const auto& segment : qd.routesContainingStop(q)) {
                if (segment.stopIndex == StopIndex(0)) continue; // Can't transfer from the first stop of a trip
                sources.push_back({segment.routeId, segment.stopIndex, footPathTime});
            }
        }

        // Sort by route to maintain excellent cache locality over contiguous CRS flat vectors
        std::ranges::sort(sources.begin(), sources.end(), [](const SourceSegment& a, const SourceSegment& b) {
            if (a.route != b.route) return a.route < b.route;
            if (a.i != b.i) return a.i < b.i;
            return a.footPathTime < b.footPathTime;
        });

        // Scan sources and collect valid connections
        for (const auto& src : sources) {
            TripId firstTrip = qd.firstTripOfRoute[src.route];
            TripId endTrip = qd.firstTripOfRoute[src.route + 1];
            uint32_t numTrips = endTrip - firstTrip;
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

                if (arrTime > maxArr) break; // Window closed; later trips will arrive too late

                // Filter out U-Turns and same-route forward invalidities
                if (src.route == toRoute && toTrip >= t && toIndex >= src.i) continue;
                if (isUTurn(t, src.i, toTrip, toIndex)) continue;

                PersistentStopEventId pEv = queryData_->flatToPersistentEvent[ev];
                if (pEv.isValid()) {
                    out.push_back(pEv);
                }
            }
        }

        std::ranges::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

    /// Expand a stop into itself + footpath neighbors with transfer time.
    inline void appendConnectedStops(StopId fromStop, std::vector<std::pair<StopId, Time>>& out) const {
        out.emplace_back(fromStop, 0);

        const auto& tg = queryData_->queryData.transferGraph;
        for (const auto edge : tg.edgesFrom(fromStop)) {
            auto toStop = StopId(tg.get(ToVertex, edge));
            Time travelTime = Time(tg.get(TravelTime, edge));
            out.emplace_back(toStop, travelTime);
        }
    }

    // === Diff / apply ===

    /// Apply the diff between current outgoing edges and the desired set
    /// using an outgoing batch (Direction::Outgoing).
    /// Assumes store_.outgoing_sorted(fromEvent) is sorted by `to` and unique.
    /// Persist TransferMeta for unchanged edges; new edges get isMinimized=false.
    /// Any add/remove marks the source trip for re-minimization.
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

    /// Apply the diff between current incoming edges and the desired set
    /// using an incoming batch (Direction::Incoming).
    /// Persist TransferMeta for unchanged edges; new edges get isMinimized=false.
    /// When a new edge is inserted, trigger domination cleanup.
    /// If a removed edge had isMinimized=true, mark the source trip for re-minimization.
    void applyIncomingDiff(PersistentStopEventId toEvent, std::span<const PersistentStopEventId> desired) const {
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
                ++curr_it;
            } else if (currentFrom > desiredFrom) {
                store_.add_incoming_edge(batch, desiredFrom, TransferMeta{false});

                // 2. Track the insertion instead of calling dominationCleanup immediately
                newlyInserted.push_back(desiredFrom);
                ++des_it;
            } else {
                ++curr_it;
                ++des_it;
            }
        }

        while (curr_it != current_span.end()) {
            // TransferMeta meta = store_.readEdgeMeta(currentFrom, toEvent);
            store_.remove_incoming_edge(batch, *curr_it);
            // if (meta.isMinimized) {
            //     StopEventId flatFromEvent = queryData_->persistentToFlatEvent[currentFrom];
            //     TripId flatFromTrip = queryData_->queryData.tripOfStopEvent[flatFromEvent];
            //     PersistentTripId pFromTrip = queryData_->flatToPersistentTrip[flatFromTrip];
            //     // TODO: Mark source trip (pFromTrip) for re-minimization
            // }
            ++curr_it;
        }

        while (des_it != desired.end()) {
            store_.add_incoming_edge(batch, *des_it, TransferMeta{false});
            newlyInserted.push_back(*des_it);
            ++des_it;
        }

        // 3. Safely commit all incoming operations first
        store_.commit_batch(batch);

        // 4. Safely execute the nested outgoing batches for cleanup
        for (const auto& fromEvent : newlyInserted) {
            dominationCleanupForInsertedTransfer(fromEvent, toEvent);
        }
    }

    // === Structural removals and redirections ===

    /// Clear all outgoing and incoming transfers for a list of stop events.
    void clearTripTransfers(std::span<const PersistentStopEventId> events) const {
        for (const PersistentStopEventId event : events) {
            clearEventTransfers(event);
        }
    }

    /// Redirect all incoming transfers of a cancelled trip to the next feasible active trip of the route.
    /// Assumes the cancelled trip's incoming edges are still present when called.
    void redirectIncomingTransfers(const DynamicTimeTable::CancelledTripInfo& trip) const {
        if (trip.eventsOfCancelledTrips.empty()) return;

        RouteId targetRoute = queryData_->persistentToFlatRoute[trip.oldRouteId];
        if (targetRoute == noRouteId) return;

        for (std::size_t i = 0; i < trip.eventsOfCancelledTrips.size(); ++i) {
            const PersistentStopEventId cancelledEvent = trip.eventsOfCancelledTrips[i];
            const StopIndex stopIndex(static_cast<uint32_t>(i));

            const auto incoming = store_.incoming_sorted(cancelledEvent);
            if (incoming.empty()) continue;

            for (const NodeID from : incoming) {
                StopEventId flatFromEvent = queryData_->persistentToFlatEvent[from];
                if (flatFromEvent == noStopEvent) continue;

                // 1. Calculate transfer window
                Time arrTime = arrivalTimeOfEvent(from);
                StopId fromStop = stopOfEvent(from);

                // Reconstruct the StopId of the cancelled event for this index
                StopId targetStop =
                    queryData_->queryData
                        .routeStopSequences[queryData_->queryData.firstStopIdOfRoute[targetRoute] + stopIndex];

                // Compute minimum departure (Arrival + Transfer duration)
                Time transferTime = getTransferDuration(fromStop, targetStop);
                if (transferTime == noTime) continue;
                Time minDepTime = arrTime + transferTime;

                // 2. Discover the true earliest feasible trip on the original route
                // This replaces the reliance on nextActiveTrip.
                std::optional<TripId> optTargetTrip = findEarliestTripOnRoute(targetRoute, stopIndex, minDepTime);
                if (!optTargetTrip) continue;

                TripId targetTrip = *optTargetTrip;
                TripId flatFromTrip = queryData_->queryData.tripOfStopEvent[flatFromEvent];
                RouteId fromRoute = queryData_->queryData.routeOfTrip[flatFromTrip];

                StopIndex fromIndex = stopIndexOfEvent(from);

                // 1. Same-route forward check (Strictly mirrors computeOutgoingTransfers)
                if (fromRoute == targetRoute && targetTrip >= flatFromTrip && stopIndex >= fromIndex) {
                    continue;
                }

                // 2. U-Turn prevention (Strictly mirrors computeOutgoingTransfers)
                if (isUTurn(flatFromTrip, fromIndex, targetTrip, stopIndex)) {
                    continue;
                }

                // 3. Map back to persistent space
                PersistentTripId pTargetTrip = queryData_->flatToPersistentTrip[targetTrip];
                std::optional<PersistentStopEventId> redirectEvent = eventId(pTargetTrip, stopIndex);

                if (redirectEvent) {
                    store_.add_edge(from, *redirectEvent, TransferMeta{false});
                }
            }
        }
    }

    [[nodiscard]] inline  Time getTransferDuration(const StopId from, const StopId to) const {
        if (from == to) return Time(0);
        const auto& tg = queryData_->queryData.transferGraph;
        for (const auto edge : tg.edgesFrom(from)) {
            if (StopId(tg.get(ToVertex, edge)) == to) {
                return Time(tg.get(TravelTime, edge));
            }
        }
        return noTime;
    }

    /// Clear all transfers for a single stop event (incoming + outgoing).
    inline void clearEventTransfers(PersistentStopEventId event) const {
        auto out = store_.outgoing_sorted(event);
        if (!out.empty()) {
            auto batch = store_.begin_batch(event, Store::Direction::Outgoing);
            for (const auto& edge : out) {
                store_.remove_outgoing_edge(batch, edge.to);
            }
            store_.commit_batch(batch);
        }

        auto in = store_.incoming_sorted(event);
        if (!in.empty()) {
            auto batch = store_.begin_batch(event, Store::Direction::Incoming);
            for (const auto from : in) {
                store_.remove_incoming_edge(batch, from);
            }
            store_.commit_batch(batch);
        }
    }

    // === Domination cleanup (incoming discovery only) ===

    /// Cleanup is applied via the store and reconciled at sync_barrier().
    /// If a removed edge had isMinimized=true, the source trip must be re-minimized.
    ///
    /// For a newly inserted transfer (from -> to), remove transfers from the same source event
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
    void dominationCleanupForInsertedTransfer(PersistentStopEventId fromEvent, PersistentStopEventId toEvent) const {
        StopEventId flatToEvent = queryData_->persistentToFlatEvent[toEvent];
        if (flatToEvent == noStopEvent) return;

        TripId flatToTrip = queryData_->queryData.tripOfStopEvent[flatToEvent];
        RouteId toRoute = queryData_->queryData.routeOfTrip[flatToTrip];
        StopIndex toIndex = stopIndexOfEvent(toEvent);

        // // Resolve the source trip in case we need to mark it for re-minimization
        // StopEventId flatFromEvent = queryData_->persistentToFlatEvent[fromEvent];
        // TripId flatFromTrip = queryData_->queryData.tripOfStopEvent[flatFromEvent];
        // PersistentTripId pFromTrip = queryData_->flatToPersistentTrip[flatFromTrip];

        // Open a batch for the source event's outgoing edges
        auto batch = store_.begin_batch(fromEvent, Store::Direction::Outgoing);
        auto outgoingEdges = store_.outgoing_sorted(fromEvent);

        for (const auto& edge : outgoingEdges) {
            PersistentStopEventId u2Event = edge.to;
            StopEventId flatU2Event = queryData_->persistentToFlatEvent[u2Event];
            if (flatU2Event == noStopEvent) continue;

            TripId flatU2Trip = queryData_->queryData.tripOfStopEvent[flatU2Event];
            RouteId u2Route = queryData_->queryData.routeOfTrip[flatU2Trip];

            if (u2Route == toRoute) {
                StopIndex u2Index = stopIndexOfEvent(u2Event);
                bool isDominated = false;

                // 1. Temporal Pruning (ALWAYS ON)
                // Existing edge is only dominated if it goes to a later trip AT THE SAME STOP INDEX
                if (u2Index == toIndex && flatU2Trip > flatToTrip) {
                    isDominated = true;
                }
                // 2. Spatial Pruning (OPTIONAL)
                else if constexpr (kEnableSpacialPruning) {
                    if (flatU2Trip == flatToTrip && flatU2Event > flatToEvent) {
                        isDominated = true;
                    }
                }

                if (isDominated) {
                    store_.remove_outgoing_edge(batch, u2Event);

                    if (edge.meta.isMinimized) {
                        // TODO: Mark source trip (pFromTrip) for re-minimization.
                    }
                }
            }
        }

        store_.commit_batch(batch);
    }
    // === Minimization (flag updates only; full set remains intact) ===

    /// Clear all minimization flags for a trip.
    void clearMinimizationFlags(PersistentTripId trip);

    /// Re-run minimization for a trip and set TransferMeta::isMinimized flags.
    void recomputeMinimizedForTrip(PersistentTripId trip);

    /// Re-run minimization for all trips on a route (loop wrapper).
    void recomputeMinimizedForRoute(PersistentRouteId route);

    // === Transfer validity helpers ===

    /// Guard against U-turn transfers that are invalid by topology/time.
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

    /// Map (trip, index) -> stop event id (if valid).
    [[nodiscard]] inline std::optional<PersistentStopEventId> eventId(PersistentTripId trip, StopIndex index) const {
        TripId flatTrip = queryData_->persistentToFlatTrip[trip];
        if (flatTrip == noTripId) return std::nullopt;
        StopEventId firstEvent = queryData_->queryData.firstStopEventOfTrip[flatTrip];
        StopEventId flatEvent = StopEventId(firstEvent + index);
        PersistentStopEventId pEvent = queryData_->flatToPersistentEvent[flatEvent];
        return pEvent;
    }

    /// Resolve stop index from a stop event id.
    [[nodiscard]] inline StopIndex stopIndexOfEvent(PersistentStopEventId event) const {
        StopEventId flatEvent = queryData_->persistentToFlatEvent[event];
        TripId flatTrip = queryData_->queryData.tripOfStopEvent[flatEvent];
        StopEventId firstEvent = queryData_->queryData.firstStopEventOfTrip[flatTrip];
        return StopIndex(flatEvent - firstEvent);
    }

    /// Resolve stop id from a stop event id.
    [[nodiscard]] inline StopId stopOfEvent(PersistentStopEventId event) const {
        StopEventId flatEvent = queryData_->persistentToFlatEvent[event];
        return queryData_->queryData.eventLookup[flatEvent].stop;
    }

    /// Access arrival time from a stop event id.
    [[nodiscard]] inline Time arrivalTimeOfEvent(PersistentStopEventId event) const {
        StopEventId flatEvent = queryData_->persistentToFlatEvent[event];
        return Time(queryData_->queryData.eventArrTimes[flatEvent]);
    }

    /// Access departure time from a stop event id.
    [[nodiscard]] inline Time departureTimeOfEvent(PersistentStopEventId event) const {
        StopEventId flatEvent = queryData_->persistentToFlatEvent[event];
        return Time(queryData_->queryData.eventDepTimes[flatEvent]);
    }

    // === Query-data access ===

    /// Earliest feasible trip on a flat route for a given stop index and time.
    [[nodiscard]] inline std::optional<TripId> findEarliestTripOnRoute(RouteId route, StopIndex stopIndex,
                                                                       Time minDepartureTime) const {
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
            Time dep = Time(routeLabel.departureTimes[baseOffset + static_cast<size_t>(mid)]);
            if (dep >= minDepartureTime) {
                bestTrip = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        if (bestTrip != -1) {
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
