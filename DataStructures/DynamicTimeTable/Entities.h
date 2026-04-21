#pragma once

#include "../../Helpers/IO/Serialization.h"
#include "../../Helpers/Types.h"

#include <vector>
#include <cstdint>

namespace DynamicTimeTable {

// ---------------------------------------------------------
// Core Internal Entities
// ---------------------------------------------------------

struct DynamicStopEvent {
    StopEventId id;           // STABLE: Never changes once minted
    StopId stop;
    uint32_t arrivalTime;
    uint32_t departureTime;

    // Additional flags for constraints (e.g., no pickup/dropoff) can be added here
    bool isSkipped = false; 

    inline void serialize(IO::Serialization& serialize) const noexcept {
        serialize(id, stop, arrivalTime, departureTime, isSkipped);
    }

    inline void deserialize(IO::Deserialization& deserialize) noexcept {
        deserialize(id, stop, arrivalTime, departureTime, isSkipped);
    }
};

struct DynamicTrip {
    std::vector<DynamicStopEvent> stopEvents;
    bool isActive = true; // False if cancelled

    // We can identify a trip externally by its first StopEventId 
    StopEventId getFirstEventId() const {
        return stopEvents.empty() ? noStopEvent : stopEvents.front().id;
    }

    inline void serialize(IO::Serialization& serialize) const noexcept {
        serialize(stopEvents, isActive);
    }

    inline void deserialize(IO::Deserialization& deserialize) noexcept {
        deserialize(stopEvents, isActive);
    }
};

struct DynamicRoute {
    RouteId routeId;
    std::vector<StopId> stopSequence;
    
    // Trips belonging to this route. 
    // Always maintained in strictly sorted order by first departure time (FIFO).
    std::vector<DynamicTrip> trips; 

    inline void serialize(IO::Serialization& serialize) const noexcept {
        serialize(routeId, stopSequence, trips);
    }

    inline void deserialize(IO::Deserialization& deserialize) noexcept {
        deserialize(routeId, stopSequence, trips);
    }
};

} // namespace DynamicTimeTable