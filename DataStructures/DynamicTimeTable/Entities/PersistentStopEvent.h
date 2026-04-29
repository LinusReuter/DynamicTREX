#pragma once

#include "../../../Helpers/IO/Serialization.h"
#include "../../../Helpers/Types.h"

namespace DynamicTimeTable {

struct PersistentStopEvent {
    StopId stop = noStop;

    Time arrivalTime = noTime;
    Time departureTime = noTime;

    bool isSkipped = false;

    inline void serialize(IO::Serialization& s) const noexcept { s(stop, arrivalTime, departureTime, isSkipped); }

    inline void deserialize(IO::Deserialization& d) noexcept { d(stop, arrivalTime, departureTime, isSkipped); }
};

}  // namespace DynamicTimeTable
