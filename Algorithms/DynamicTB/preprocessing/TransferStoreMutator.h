#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "../../../Helpers/Types.h"
#include "../../../Helpers/UpdateCounters.h"
#include "../../DynamicTimeTable/BuildQueryData.h"
#include "AffectedEventSink.h"
#include "TransferTypes.h"

namespace DynamicTB::Preprocessing {

/**
 * @brief The only layer that mutates the transfer store.
 *
 * Everything here diffs a freshly discovered edge set (from TransferDiscovery) against what
 * the store currently holds and applies the difference through the store's batch API. It
 * also produces the two downstream signals:
 *  - re-minimization targets (`MinTarget`), consumed by TransferMinimizer;
 *  - level-0 affected events (`AffectedSink`), consumed by the TREX customization.
 *
 * Templated on the store type so the per-edge store calls devirtualize; templated on the
 * sink so the affected-set marking disappears entirely when TREX is not attached.
 */
template <class Store, class AffectedSink = NullAffectedSink>
class TransferStoreMutator {
public:
    using DynamicQueryData = DynamicTimeTable::Algo::DynamicQueryData;
    using NodeID = PersistentStopEventId;
    using OutEdge = typename Store::OutEdge;

    TransferStoreMutator(Store& store, const DynamicQueryData& queryData) noexcept
        : store_(store), queryData_(&queryData) {}

    inline void setQueryData(const DynamicQueryData& queryData) noexcept { queryData_ = &queryData; }

    /**
     * @brief Record the (flat) source trip of a stop event for later re-minimization.
     * Converts persistent -> flat exactly once at the store boundary; inactive events are ignored.
     */
    inline void recordSourceTripOfEvent(PersistentStopEventId from, std::vector<MinTarget>& localTrips) const {
        const StopEventId flatEvent = queryData_->persistentToFlatEvent[from];
        if (flatEvent == noStopEvent) return;
        // Warm-start boundary = the stop index at which this source boards. A change to its edge
        // into a downstream event only affects keep-decisions at this stop and earlier ones.
        localTrips.emplace_back(queryData_->tripOfEvent(flatEvent), queryData_->stopIndexOfEvent(flatEvent));
    }

    /**
     * @brief Two-way sorted set-difference merge of a current edge list against a desired one.
     * `key` projects a current element to its PersistentStopEventId. The callbacks fire on
     * entries present only in `current` (onRemove), only in `desired` (onAdd), and in both
     * (onKeep). Both inputs must be sorted ascending by PersistentStopEventId.
     */
    template <typename CurrentRange, typename Key, typename OnRemove, typename OnAdd, typename OnKeep>
    static inline void mergeSortedDiff(CurrentRange&& current, std::span<const PersistentStopEventId> desired, Key key,
                                       OnRemove onRemove, OnAdd onAdd, OnKeep onKeep) {
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
     * @return true iff this source needs re-minimization.
     */
    inline bool applyOutgoingDiff(PersistentStopEventId fromEvent, std::span<const PersistentStopEventId> desired,
                                  TransferUpdateCounters& lc, AffectedSink& sink) const {
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
                if constexpr (collectTransferStats) ++lc.outgoingEdgesRemoved;
                if (edge.meta.isMinimized) {
                    needsRemin = true;
                    // The reduced set lost this edge: both endpoints are level-0 affected.
                    sink.markEdgeChanged(fromEvent, edge.to);
                }
                store_.remove_outgoing_edge(batch, edge.to);
            },
            [&](PersistentStopEventId to) {
                if constexpr (collectTransferStats) ++lc.outgoingEdgesAdded;
                // New edges always enter outside the reduced set; if minimization later keeps
                // one, the flip is observed there.
                store_.add_outgoing_edge(batch, to, TransferMeta{});
                needsRemin = true;
            },
            [](const auto&) {});
        store_.commit_batch(batch);
        return needsRemin;
    }

    /**
     * @brief Computes mutations against current incoming store entries.
     */
    inline void applyIncomingDiff(PersistentStopEventId toEvent, StopEventId flatToEvent,
                                  std::span<const PersistentStopEventId> desired, std::vector<MinTarget>& localTrips,
                                  std::vector<PendingDominationCleanup>& pendingCleanups,
                                  std::vector<PersistentStopEventId>& newlyInserted, TransferUpdateCounters& lc,
                                  AffectedSink& sink) const {
        auto batch = store_.begin_batch(toEvent, Store::Direction::Incoming);

        // New incoming edges whose domination cleanup is deferred.
        // Track reduction targets on edge changes.
        newlyInserted.clear();

        mergeSortedDiff(
            store_.incoming_sorted(toEvent), desired, [](PersistentStopEventId from) { return from; },
            [&](PersistentStopEventId from) {
                if constexpr (collectTransferStats) ++lc.incomingEdgesRemoved;
                store_.remove_incoming_edge(batch, from);
                recordSourceTripOfEvent(from, localTrips);
                // Incoming storage carries no metadata, so we cannot tell whether the
                // mirrored outgoing edge was in the reduced set. Mark unconditionally:
                // over-approximating the affected set only costs extra customization work,
                // while missing an entry would leave a rank too low (i.e. wrong answers).
                sink.markEdgeChanged(from, toEvent);
            },
            [&](PersistentStopEventId from) {
                if constexpr (collectTransferStats) ++lc.incomingEdgesAdded;
                recordSourceTripOfEvent(from, localTrips);
                store_.add_incoming_edge(batch, from, TransferMeta{});
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

    /**
     * @brief If a newly inserted transfer (fromEvent -> flatToEvent) dominates an existing
     * outgoing transfer of fromEvent (same route, same stop index, but a later trip), remove it.
     */
    inline void dominationCleanupForInsertedTransfer(PersistentStopEventId fromEvent, StopEventId flatToEvent,
                                                     std::vector<MinTarget>& localTrips, TransferUpdateCounters& lc,
                                                     AffectedSink& sink) const {
        const auto& qd = queryData_->queryData;
        TripId flatToTrip = queryData_->tripOfEvent(flatToEvent);
        RouteId toRoute = qd.routeOfTrip[flatToTrip];
        StopIndex toIndex = queryData_->stopIndexOfEvent(flatToEvent);

        // Snapshot fromEvent's outgoing edges under the store lock. This runs in the
        // parallel incoming phase, where other threads mirror incoming edges into
        // out_[fromEvent] and may reallocate it; a raw span would dangle (heap UAF).
        thread_local std::vector<OutEdge> outgoingSnapshot;
        store_.copy_outgoing(fromEvent, outgoingSnapshot);

        for (const auto& edge : outgoingSnapshot) {
            PersistentStopEventId u2Event = edge.to;
            StopEventId flatU2Event = queryData_->persistentToFlatEvent[u2Event];
            if (flatU2Event == noStopEvent) continue;
            if (queryData_->routeOfEvent(flatU2Event) != toRoute) continue;

            const bool isDominated = queryData_->stopIndexOfEvent(flatU2Event) == toIndex &&
                                     queryData_->tripOfEvent(flatU2Event) > flatToTrip;
            if (!isDominated) continue;

            store_.remove_edge(fromEvent, u2Event);
            if constexpr (collectTransferStats) ++lc.dominationEdgesRemoved;
            if (edge.meta.isMinimized) {
                recordSourceTripOfEvent(fromEvent, localTrips);
                sink.markEdgeChanged(fromEvent, u2Event);
            }
            break;
        }
    }

private:
    Store& store_;
    const DynamicQueryData* queryData_{nullptr};
};

}  // namespace DynamicTB::Preprocessing
