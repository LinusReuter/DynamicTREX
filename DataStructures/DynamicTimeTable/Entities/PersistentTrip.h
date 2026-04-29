#pragma once

#include <cstdint>

#include "../../../Helpers/IO/Serialization.h"
#include "../../../Helpers/Types.h"

namespace DynamicTimeTable {

struct PersistentTrip {
    // Current route assignment in the persistent workspace.
    PersistentRouteId route = noPersistentRouteId;

    // First event of the trip's minted contiguous event block.
    PersistentStopEventId firstEvent = noPersistentStopEventId;

    // Total minted stop events in the block (including skipped ones).
    std::uint32_t numberOfEvents = 0;

    // False if the trip is cancelled.
    bool isActive = true;

    inline void serialize(IO::Serialization& s) const noexcept { s(route, firstEvent, numberOfEvents, isActive); }

    inline void deserialize(IO::Deserialization& d) noexcept { d(route, firstEvent, numberOfEvents, isActive); }
};

}  // namespace DynamicTimeTable
