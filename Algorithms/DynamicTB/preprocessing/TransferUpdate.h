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
    using Store = ITansferStore<PersistentStopEventId, TransferMeta>;
    using DynamicQueryData = DynamicTimeTable::Algo::DynamicQueryData;

    explicit TransferUpdate(Store& store);

    /**
     * Full rebuild: clears the store and (re)discovers all outgoing transfers.
     *
     * Implementation notes:
     * - Ensure the store has nodes for all stop events (store.add_nodes(maxEventId)).
     * - Allow temporary inconsistency during bulk inserts, then rebuild/sync incoming once.
     *
     * This does NOT set minimization flags
     * (call buildInitialMinimizedTransfers).
     */
    void buildInitialFullTransfers(const DynamicQueryData& queryData);

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
    void updateOutgoingForEvent(PersistentStopEventId event);

    /// Update (compute + apply) incoming transfers for a single stop event.
    /// Uses store.incoming(target) for the current set and applies diffs via incoming batches.
    /// Requires a valid departure time (constraints are modeled as invalid times),
    /// and must enforce U-turn filtering + max-wait cap.
    /// New edges get isMinimized=false; existing edges preserve metadata.
    /// When new transfers are inserted, trigger domination cleanup on the target line.
    void updateIncomingForEvent(PersistentStopEventId event);

    /// Compute all feasible outgoing transfers from a single stop event.
    /// Skips events with invalid arrival times (constraints modeled as invalid times).
    void computeOutgoingTransfers(PersistentStopEventId fromEvent,
                                  std::vector<PersistentStopEventId>& out) const;

    /// Compute all feasible incoming transfers to a single stop event.
    /// Skips events with invalid departure times (constraints modeled as invalid times).
    void computeIncomingTransfers(PersistentStopEventId toEvent,
                                  std::vector<PersistentStopEventId>& out) const;

    /// Expand a stop into itself + footpath neighbors with transfer time.
    void appendConnectedStops(StopId fromStop,
                              std::vector<std::pair<StopId, Time>>& out) const;

    /// Find earliest feasible event on any route containing the stop.
    std::optional<PersistentStopEventId> findEarliestEvent(StopId stop, Time minDepartureTime,
                                                           PersistentRouteId forbidRoute,
                                                           StopIndex minIndex) const;

    // === Diff / apply ===

    /// Apply the diff between current outgoing edges and the desired set
    /// using an outgoing batch (incoming=false).
    /// Persist TransferMeta for unchanged edges; new edges get isMinimized=false.
    /// Any add/remove marks the source trip for re-minimization.
    void applyOutgoingDiff(PersistentStopEventId fromEvent,
                           std::span<const PersistentStopEventId> desired);

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
    bool isUTurn(PersistentTripId fromTrip, StopIndex fromIndex,
                 PersistentTripId toTrip, StopIndex toIndex) const;

    // === Timetable resolution helpers ===

    /// Map (trip, index) -> stop event id (if valid).
    std::optional<PersistentStopEventId> eventId(PersistentTripId trip, StopIndex index) const;

    /// Resolve stop index from a stop event id.
    StopIndex stopIndexOfEvent(PersistentStopEventId event) const;

    /// Resolve stop id from a stop event id.
    StopId stopOfEvent(PersistentStopEventId event) const;

    /// Access arrival time from a stop event id.
    Time arrivalTimeOfEvent(PersistentStopEventId event) const;

    /// Access departure time from a stop event id.
    Time departureTimeOfEvent(PersistentStopEventId event) const;

    // === Query-data access ===

    /// Earliest feasible trip on a flat route for a given stop index + time.
    /// Must respect a max-wait cap (e.g., 24h) to avoid unrealistic transfers.
    std::optional<TripId> findEarliestTripOnRoute(RouteId route, StopIndex stopIndex, Time minDepartureTime) const;

private:
    const DynamicQueryData* queryData_{nullptr};
    Store& store_;
};

}  // namespace DynamicTB::Preprocessing
