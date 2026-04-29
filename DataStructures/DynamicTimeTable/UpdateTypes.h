#pragma once

#include <vector>

#include "../../Helpers/Types.h"

namespace DynamicTimeTable {

// ---------------------------------------------------------
// Incoming Update Inputs
// ---------------------------------------------------------

struct StopModification {
    StopIndex stopIndex;
    Time newArrivalTime = noTime;  // noValue if not affected
    Time newDepartureTime = noTime;
    bool isSkipped = false;  // If true, triggers a route extraction/re-insertion
};

struct AddedTripInfo {
    PersistentRouteId preferredRouteId = noPersistentRouteId;
    std::vector<StopId> stopSequence;
    std::vector<Time> arrivalTimes;
    std::vector<Time> departureTimes;
};

struct PendingUpdates {
    std::vector<PersistentTripId> cancellations;
    std::vector<std::pair<PersistentTripId, std::vector<StopModification>>> modifications;
    std::vector<AddedTripInfo> additions;

    bool hasUpdates() const { return !cancellations.empty() || !modifications.empty() || !additions.empty(); }
};

// ---------------------------------------------------------
// Change Tracking (Consumed by the Transfer Update Stage)
// ---------------------------------------------------------

struct CancelledTripInfo {
    PersistentTripId tripId;
    PersistentRouteId oldRouteId;  // Essential for transfer phase redirection/cleanup
};

struct ChangeSummary {
    // PHASE 0: Complete Route Removals
    // Transfer stage deletes all edges pointing to or from these Routes.
    std::vector<PersistentRouteId> removedRoutes;

    // PHASE 1: Trip Cancellations & Extractions
    // Transfer stage treats these as removed from their old route context.
    // (This includes trips that were permanently cancelled AND trips that were
    // extracted due to a skipped stop or FIFO violation).
    std::vector<CancelledTripInfo> cancelledTrips;

    // PHASE 2 & 3: Discovery Triggers - New Additions
    // Trips that were genuinely added, or trips that were re-inserted into a new
    // route after a FIFO violation / skipped stop extraction.
    std::vector<PersistentTripId> addedTrips;

    // Discovery Triggers - In-Place Modifications
    // Events that were delayed/modified but the trip stayed in its original route.
    std::vector<PersistentStopEventId> modifiedEvents;

    // MINIMIZATION: Upstream Impact
    // If a trip's arrival times were delayed, transfers pointing INTO it
    // are now worse. The Transfer Stage must look up incoming edges to this trip,
    // and flag their SOURCE trips for re-minimization.
    std::vector<PersistentTripId> tripsWithDelayedArrivals;

    void clear() {
        removedRoutes.clear();
        cancelledTrips.clear();
        addedTrips.clear();
        modifiedEvents.clear();
        tripsWithDelayedArrivals.clear();
    }

    bool hasStructuralChanges() const {
        return !removedRoutes.empty() || !cancelledTrips.empty() || !addedTrips.empty() || !modifiedEvents.empty() ||
               !tripsWithDelayedArrivals.empty();
    }
};

struct UpdateStatistics {
    std::size_t totalUpdates = 0;
    std::size_t successfulUpdates = 0;
    std::size_t failedUpdates = 0;
    std::size_t cancellations = 0;
    std::size_t modifications = 0;
    std::size_t additions = 0;

    UpdateStatistics& operator+=(const UpdateStatistics& other) {
        totalUpdates += other.totalUpdates;
        successfulUpdates += other.successfulUpdates;
        failedUpdates += other.failedUpdates;
        cancellations += other.cancellations;
        modifications += other.modifications;
        additions += other.additions;
        return *this;
    }
};

}  // namespace DynamicTimeTable
