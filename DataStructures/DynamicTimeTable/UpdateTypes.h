#pragma once

#include "../../Helpers/Types.h"

#include <vector>
#include <optional>
#include <cstdint>

namespace DynamicTimeTable {

// ---------------------------------------------------------
// Incoming Update Inputs
// ---------------------------------------------------------

struct StopModification {
    StopIndex stopIndex;
    uint32_t newArrivalTime;
    uint32_t newDepartureTime;
    bool isSkipped = false;
};

struct AddedTripInfo {
    std::optional<RouteId> preferredRouteId;
    std::vector<StopId> stopSequence;
    std::vector<uint32_t> arrivalTimes;
    std::vector<uint32_t> departureTimes;
};

struct PendingUpdates {
    // Identify trips to cancel or modify by their first StopEventId
    std::vector<StopEventId> cancellations;
    std::vector<std::pair<StopEventId, std::vector<StopModification>>> modifications;
    std::vector<AddedTripInfo> additions;

    bool hasUpdates() const {
        return !cancellations.empty() || !modifications.empty() || !additions.empty();
    }
};

// ---------------------------------------------------------
// Change Tracking (Consumed by the Transfer Update Stage)
// ---------------------------------------------------------

struct ChangeSummary {
    // PHASE 0: Complete Line Removals
    // Transfer stage deletes all edges pointing to or from these Routes.
    std::vector<RouteId> removedRoutes;

    // PHASE 1: Trip Cancellations
    // Transfer stage attempts to redirect edges pointing to these trips.
    // (Identified by their first StopEventId to locate them in the TransferStore)
    std::vector<StopEventId> cancelledTrips;

    // PHASE 2 & 3: Discovery Triggers
    // Trips that were added, or had departure/arrival delays.
    // Transfer stage runs Outgoing/Incoming discovery on their affected stops.
    std::vector<StopEventId> modifiedOrAddedTrips;

    // MINIMIZATION: Upstream Impact
    // If a trip's arrival times were delayed, transfers pointing INTO it
    // are now worse. The Transfer Stage must look up incoming edges to this trip,
    // and flag their SOURCE trips for re-minimization.
    std::vector<StopEventId> tripsWithDelayedArrivals;

    void clear() {
        removedRoutes.clear();
        cancelledTrips.clear();
        modifiedOrAddedTrips.clear();
        tripsWithDelayedArrivals.clear();
    }

    bool hasStructuralChanges() const {
        return !removedRoutes.empty() || !cancelledTrips.empty() ||
               !modifiedOrAddedTrips.empty() || !tripsWithDelayedArrivals.empty();
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

} // namespace DynamicTimeTable
