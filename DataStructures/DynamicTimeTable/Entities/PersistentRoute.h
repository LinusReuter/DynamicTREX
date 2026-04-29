#pragma once

#include <vector>

#include "../../../Helpers/IO/Serialization.h"
#include "../../../Helpers/Types.h"

namespace DynamicTimeTable {

/**
 * PersistentRoute
 *
 * Semantics:
 * - A route owns an immutable stop sequence (once created).
 * - Trips that change their effective stop sequence (e.g. skipped stops) must migrate
 *   to a different PersistentRouteId (existing compatible or newly created).
 *
 */
struct PersistentRoute {
    PersistentRouteId routeId = noPersistentRouteId;

    // Immutable after creation (route identity).
    std::vector<StopId> stopSequence;

    // Membership list of trips currently assigned to this route.
    // Intended to be maintained in chronological order and FIFO-clean by update logic.
    std::vector<PersistentTripId> trips;

    inline void serialize(IO::Serialization& s) const noexcept { s(routeId, stopSequence, trips); }

    inline void deserialize(IO::Deserialization& d) noexcept { d(routeId, stopSequence, trips); }
};

}  // namespace DynamicTimeTable
