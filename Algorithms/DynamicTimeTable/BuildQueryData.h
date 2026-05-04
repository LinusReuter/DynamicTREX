#pragma once

#include "../TripBased/Query/Types.h"
#include "../../DataStructures/DynamicTimeTable/Data.h"
#include "../../Helpers/Types.h"

namespace DynamicTimeTable {
namespace Algo {

struct DynamicQueryData {
    TripBased::QueryData queryData;

    // Translation Layers: Flat ID -> Persistent ID
    std::vector<PersistentRouteId> flatToPersistentRoute;
    std::vector<PersistentTripId> flatToPersistentTrip;
    std::vector<PersistentStopEventId> flatToPersistentEvent;

    // Reverse Translation Layers: Persistent ID -> Flat ID
    std::vector<RouteId> persistentToFlatRoute;
    std::vector<TripId> persistentToFlatTrip;
    std::vector<StopEventId> persistentToFlatEvent;

    static DynamicQueryData buildFromDynamic(const DynamicTimeTable::Data& data) {
        const auto& routes = data.routes();
        const auto& trips = data.trips();
        const auto& events = data.events();

        // 1. Prefix-Sum Offset Pass (Sequential, enables parallelization later)
        struct RouteOffsets {
            RouteId flatRouteId;
            TripId flatTripId;
            StopEventId flatEventId;
            size_t flatStopSeqOffset;
        };
        std::vector<RouteOffsets> routeOffsets(routes.size());

        size_t activeRouteCount = 0;
        size_t activeTripCount = 0;
        size_t activeEventCount = 0;
        size_t activeStopSequenceLength = 0;

        for (size_t i = 0; i < routes.size(); ++i) {
            routeOffsets[i] = { RouteId(activeRouteCount), TripId(activeTripCount), StopEventId(activeEventCount), activeStopSequenceLength };

            const auto& pRoute = routes[i];
            if (pRoute.trips.empty()) continue;

            activeRouteCount++;
            activeTripCount += pRoute.trips.size();
            activeEventCount += pRoute.trips.size() * pRoute.stopSequence.size();
            activeStopSequenceLength += pRoute.stopSequence.size();
        }

        // 2. Allocate Builder and Translation Layers
        TripBased::QueryDataBuilder builder;

        builder.transferGraph = data.transferGraph();
        builder.reverseTransferGraph = data.transferGraph();
        builder.reverseTransferGraph.revert();

        builder.eventLookup.resize(activeEventCount);
        builder.eventArrTimes.resize(activeEventCount);
        builder.eventDepTimes.resize(activeEventCount);

        builder.tripOfStopEvent.resize(activeEventCount);
        builder.routeOfTrip.resize(activeTripCount);
        builder.firstStopEventOfTrip.resize(activeTripCount);
        builder.firstTripOfRoute.resize(activeRouteCount + 1);

        builder.firstStopIdOfRoute.resize(activeRouteCount + 1);
        builder.routeStopSequences.resize(activeStopSequenceLength);
        builder.routeLabels.resize(activeRouteCount);

        std::vector<PersistentRouteId> flatToPersistentRoute(activeRouteCount);
        std::vector<PersistentTripId> flatToPersistentTrip(activeTripCount);
        std::vector<PersistentStopEventId> flatToPersistentEvent(activeEventCount);

        std::vector<RouteId> persistentToFlatRoute(routes.size(), (RouteId)-1);
        std::vector<TripId> persistentToFlatTrip(trips.size(), (TripId)-1);
        std::vector<StopEventId> persistentToFlatEvent(events.size(), (StopEventId)-1);

        // 3. Topology & Translation Mapping (Parallelized)
#pragma omp parallel for
        for (size_t rIdx = 0; rIdx < routes.size(); ++rIdx) {
            const auto& pRoute = routes[rIdx];
            if (pRoute.trips.empty()) continue;

            RouteId currentRoute = routeOffsets[rIdx].flatRouteId;
            TripId currentTrip = routeOffsets[rIdx].flatTripId;
            StopEventId currentEvent = routeOffsets[rIdx].flatEventId;
            size_t currentStopSeqOffset = routeOffsets[rIdx].flatStopSeqOffset;

            flatToPersistentRoute[currentRoute] = PersistentRouteId(rIdx);
            persistentToFlatRoute[rIdx] = currentRoute;

            builder.firstTripOfRoute[currentRoute] = currentTrip;
            builder.firstStopIdOfRoute[currentRoute] = currentStopSeqOffset;

            for (size_t stopIdx = 0; stopIdx < pRoute.stopSequence.size(); ++stopIdx) {
                builder.routeStopSequences[currentStopSeqOffset + stopIdx] = pRoute.stopSequence[stopIdx];
            }

            builder.routeLabels[currentRoute].numberOfTrips = pRoute.trips.size();
            if (!pRoute.stopSequence.empty()) {
                builder.routeLabels[currentRoute].departureTimes.resize((pRoute.stopSequence.size() - 1) * pRoute.trips.size());
            }

            size_t tripOffset = 0;
            for (const auto pTripId : pRoute.trips) {
                flatToPersistentTrip[currentTrip] = pTripId;
                persistentToFlatTrip[pTripId] = currentTrip;

                builder.firstStopEventOfTrip[currentTrip] = currentEvent;
                builder.routeOfTrip[currentTrip] = currentRoute;

                const auto& pTrip = trips[pTripId];
                size_t stopIdx = 0;
                for (size_t i = 0; i < pTrip.numberOfEvents; ++i) {
                    const auto eventId = PersistentStopEventId(pTrip.firstEvent + i);
                    const auto& pEvent = events[eventId];

                    if (pEvent.isSkipped) continue; // Skipped event

                    flatToPersistentEvent[currentEvent] = eventId;
                    persistentToFlatEvent[eventId] = currentEvent;

                    builder.tripOfStopEvent[currentEvent] = currentTrip;

                    builder.eventLookup[currentEvent] = TripBased::EventLookup(pRoute.stopSequence[stopIdx], pEvent.arrivalTime);
                    builder.eventArrTimes[currentEvent] = pEvent.arrivalTime;
                    builder.eventDepTimes[currentEvent] = pEvent.departureTime;

                    if (stopIdx + 1 < pRoute.stopSequence.size()) {
                        size_t labelIdx = (stopIdx * pRoute.trips.size()) + tripOffset;
                        builder.routeLabels[currentRoute].departureTimes[labelIdx] = pEvent.departureTime;
                    }

                    currentEvent++;
                    stopIdx++;
                }
                currentTrip++;
                tripOffset++;
            }
        }
        // Set final sentinel bounds natively outside the loop
        builder.firstTripOfRoute[activeRouteCount] = TripId(activeTripCount);
        builder.firstStopIdOfRoute[activeRouteCount] = activeStopSequenceLength;

        // 4. Rebuild routeSegments
        const size_t numStops = data.numberOfStops();
        std::vector<size_t> segmentsPerStop(numStops, 0);

        for (RouteId r = RouteId(0); r < RouteId(activeRouteCount); ++r) {
            const size_t start = builder.firstStopIdOfRoute[r];
            const size_t end = builder.firstStopIdOfRoute[r + 1];
            for (size_t i = start; i < end; ++i) {
                segmentsPerStop[builder.routeStopSequences[i]]++;
            }
        }

        builder.firstRouteSegmentOfStop.resize(numStops + 1, 0);
        size_t totalSegments = 0;
        for (size_t i = 0; i < numStops; ++i) {
            builder.firstRouteSegmentOfStop[i] = totalSegments;
            totalSegments += segmentsPerStop[i];
        }
        builder.firstRouteSegmentOfStop[numStops] = totalSegments;

        builder.routeSegments.resize(totalSegments);
        std::vector<size_t> currentSegmentOffset = builder.firstRouteSegmentOfStop;

        for (RouteId r = RouteId(0); r < RouteId(activeRouteCount); ++r) {
            const size_t start = builder.firstStopIdOfRoute[r];
            const size_t end = builder.firstStopIdOfRoute[r + 1];
            for (StopIndex stopIdx = StopIndex(0); start + stopIdx < end; ++stopIdx) {
                StopId stop = builder.routeStopSequences[start + stopIdx];
                RAPTOR::RouteSegment segment;
                segment.routeId = r;
                segment.stopIndex = stopIdx;

                size_t offset = currentSegmentOffset[stop]++;
                builder.routeSegments[offset] = segment;
            }
        }

        return DynamicQueryData{
            TripBased::QueryData(std::move(builder)),
            std::move(flatToPersistentRoute),
            std::move(flatToPersistentTrip),
            std::move(flatToPersistentEvent),
            std::move(persistentToFlatRoute),
            std::move(persistentToFlatTrip),
            std::move(persistentToFlatEvent)
        };
    }
};

} // namespace Algo
} // namespace DynamicTimeTable
