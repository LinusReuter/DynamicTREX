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
    std::vector<std::pair<PersistentTripId, std::vector<StopModification>>> modifications; //Sorted by stop sequence
    std::vector<AddedTripInfo> additions;

    bool hasUpdates() const { return !cancellations.empty() || !modifications.empty() || !additions.empty(); }
};

// ---------------------------------------------------------
// Change Tracking (Consumed by the Transfer Update Stage)
// ---------------------------------------------------------

struct CancelledTripInfo {
    PersistentTripId tripId;
    // Snapshot of previously ACTIVE stop events (non-skipped), in stop-index order.
    // The index in this vector corresponds to the stop index used by transfer updates.
    std::vector<PersistentStopEventId> eventsOfCancelledTrips;
};

struct ChangeSummary {
    // PHASE 0/1: Trip Cancellations & Extractions
    // Transfer stage treats these as removed from their old route context.
    // (This includes trips that were permanently cancelled AND trips that were
    // extracted due to a skipped stop or FIFO violation).
    std::vector<CancelledTripInfo> cancelledTrips;

    // Set of Trips needed Incoming rediscovery due to one or more directly previous trips were canceled
    std::vector<PersistentTripId> tripsToRediscoverIncomingDueToCancellation;

    // PHASE 2 & 3: Discovery Triggers - New Additions
    // Trips that were genuinely added, or trips that were re-inserted into a new
    // route after a FIFO violation / skipped stop extraction.
    std::vector<PersistentTripId> addedTrips;

    // Discovery Triggers - In-Place Modifications
    // Events that were delayed/modified but the trip stayed in its original route.
    // Paired with bool marking departure has negative delay (earlier)
    std::vector<std::pair<PersistentStopEventId, bool>> modifiedEvents;

    // MINIMIZATION: Upstream Impact
    // If a trip's arrival times CHANGED (later OR earlier), transfers pointing INTO it
    // may need re-minimization. The Transfer Stage looks up incoming edges to this trip
    // and flags their SOURCE trips.
    //
    // Only sources boarding at a stop index STRICTLY BEFORE some changed arrival are
    // affected (minimization dominance reads arrival times of stops after the boarding
    // stop). We therefore carry the maximum stop index whose arrival changed: incoming
    // edges into events at stop index >= maxChangedArrivalIndex can be skipped.
    std::vector<std::pair<PersistentTripId, StopIndex>> tripsWithChangedArrivals;

    void clear() {
        cancelledTrips.clear();
        addedTrips.clear();
        modifiedEvents.clear();
        tripsWithChangedArrivals.clear();
    }

    bool hasStructuralChanges() const {
        return !cancelledTrips.empty() || !addedTrips.empty() || !modifiedEvents.empty() ||
               !tripsWithChangedArrivals.empty();
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
