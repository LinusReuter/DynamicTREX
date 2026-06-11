#pragma once

#include <sstream>
#include <string>
#include <utility>

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

    // --- Query helpers (flat IDs) ---
    inline size_t numberOfStopsInTrip(const TripId trip) const noexcept {
        const auto& first = queryData.firstStopEventOfTrip;
        const size_t tripIdx = static_cast<size_t>(trip);
        const size_t start = static_cast<size_t>(first[tripIdx]);
        const size_t end = (tripIdx + 1 < first.size()) ? static_cast<size_t>(first[tripIdx + 1])
                                                        : queryData.eventLookup.size();
        return end - start;
    }

    inline StopEventId stopEventIdOfTripStop(const TripId trip, const StopIndex index) const noexcept {
        return StopEventId(queryData.firstStopEventOfTrip[trip] + index);
    }

    inline StopId getStop(const TripId trip, const StopIndex index) const noexcept {
        const StopEventId event = stopEventIdOfTripStop(trip, index);
        return queryData.eventLookup[event].stop;
    }

    inline Time arrivalTime(const TripId trip, const StopIndex index) const noexcept {
        const StopEventId event = stopEventIdOfTripStop(trip, index);
        return Time(queryData.eventArrTimes[event]);
    }

    inline Time departureTime(const TripId trip, const StopIndex index) const noexcept {
        const StopEventId event = stopEventIdOfTripStop(trip, index);
        return Time(queryData.eventDepTimes[event]);
    }

    inline std::vector<PersistentStopEventId> getEventsOfTrip(const PersistentTripId pTrip) const noexcept {
        TripId trip = persistentToFlatTrip[pTrip];
        std::vector<PersistentStopEventId> pEvents;
        if (trip == noTripId) return pEvents;
        StopEventId first = queryData.firstStopEventOfTrip[trip];
        StopEventId limit = queryData.firstStopEventOfTrip[trip + 1];
        pEvents.reserve(limit - first);
        for (StopEventId event = first; event < limit; ++event) {
            pEvents.emplace_back(flatToPersistentEvent[event]);
        }
        return pEvents;
    }

    std::pair<bool, std::string> validate(const DynamicTimeTable::Data& data) const {
        std::stringstream error_msg;
        const auto& qd = queryData;

        const size_t numStops = data.numberOfStops();
        const size_t numRoutes = qd.routeLabels.size();
        const size_t numTrips = qd.routeOfTrip.size();
        const size_t numEvents = qd.eventArrTimes.size();

        if (persistentToFlatRoute.size() != data.routes().size()) {
            error_msg << "persistentToFlatRoute size mismatch: have " << persistentToFlatRoute.size()
                      << ", expected " << data.routes().size();
            return {false, error_msg.str()};
        }
        if (persistentToFlatTrip.size() != data.trips().size()) {
            error_msg << "persistentToFlatTrip size mismatch: have " << persistentToFlatTrip.size()
                      << ", expected " << data.trips().size();
            return {false, error_msg.str()};
        }
        if (persistentToFlatEvent.size() != data.events().size()) {
            error_msg << "persistentToFlatEvent size mismatch: have " << persistentToFlatEvent.size()
                      << ", expected " << data.events().size();
            return {false, error_msg.str()};
        }

        if (flatToPersistentRoute.size() != numRoutes) {
            error_msg << "flatToPersistentRoute size mismatch: have " << flatToPersistentRoute.size()
                      << ", expected " << numRoutes;
            return {false, error_msg.str()};
        }
        if (flatToPersistentTrip.size() != numTrips) {
            error_msg << "flatToPersistentTrip size mismatch: have " << flatToPersistentTrip.size() << ", expected "
                      << numTrips;
            return {false, error_msg.str()};
        }
        if (flatToPersistentEvent.size() != numEvents) {
            error_msg << "flatToPersistentEvent size mismatch: have " << flatToPersistentEvent.size()
                      << ", expected " << numEvents;
            return {false, error_msg.str()};
        }

        if (qd.eventLookup.size() != numEvents || qd.eventDepTimes.size() != numEvents ||
            qd.tripOfStopEvent.size() != numEvents) {
            error_msg << "Stop-event arrays size mismatch";
            return {false, error_msg.str()};
        }

        if (qd.firstTripOfRoute.size() != numRoutes + 1) {
            error_msg << "firstTripOfRoute size mismatch: have " << qd.firstTripOfRoute.size()
                      << ", expected " << (numRoutes + 1);
            return {false, error_msg.str()};
        }
        if (static_cast<size_t>(qd.firstTripOfRoute[numRoutes]) != numTrips) {
            error_msg << "firstTripOfRoute sentinel mismatch: got " << qd.firstTripOfRoute[numRoutes]
                      << ", expected " << numTrips;
            return {false, error_msg.str()};
        }

        if (qd.firstStopIdOfRoute.size() != numRoutes + 1) {
            error_msg << "firstStopIdOfRoute size mismatch: have " << qd.firstStopIdOfRoute.size()
                      << ", expected " << (numRoutes + 1);
            return {false, error_msg.str()};
        }
        if (qd.firstStopIdOfRoute[numRoutes] != qd.routeStopSequences.size()) {
            error_msg << "firstStopIdOfRoute sentinel mismatch: got " << qd.firstStopIdOfRoute[numRoutes]
                      << ", expected " << qd.routeStopSequences.size();
            return {false, error_msg.str()};
        }

        if (qd.firstRouteSegmentOfStop.size() != numStops + 1) {
            error_msg << "firstRouteSegmentOfStop size mismatch: have " << qd.firstRouteSegmentOfStop.size()
                      << ", expected " << (numStops + 1);
            return {false, error_msg.str()};
        }
        if (qd.firstRouteSegmentOfStop[numStops] != qd.routeSegments.size()) {
            error_msg << "routeSegments sentinel mismatch: got " << qd.firstRouteSegmentOfStop[numStops]
                      << ", expected " << qd.routeSegments.size();
            return {false, error_msg.str()};
        }

        if (qd.firstStopEventOfTrip.size() != numTrips + 1) {
            error_msg << "firstStopEventOfTrip size mismatch: have " << qd.firstStopEventOfTrip.size()
                      << ", expected " << (numTrips + 1) << " (sentinel element required for boundary queries)";
            return {false, error_msg.str()};
        }
        if (static_cast<size_t>(qd.firstStopEventOfTrip[numTrips]) != numEvents) {
            error_msg << "firstStopEventOfTrip sentinel mismatch: got " << qd.firstStopEventOfTrip[numTrips]
                      << ", expected " << numEvents;
            return {false, error_msg.str()};
        }

        for (size_t i = 0; i < qd.routeStopSequences.size(); ++i) {
            const StopId stop = qd.routeStopSequences[i];
            if (static_cast<size_t>(stop) >= numStops) {
                error_msg << "Invalid stop id " << stop << " in routeStopSequences at index " << i;
                return {false, error_msg.str()};
            }
        }

        auto adjustedArrival = [&](const StopId stop, const std::uint32_t time) -> int64_t {
            int64_t value = static_cast<int64_t>(time);
            if (data.usesImplicitArrivalBufferTimes()) {
                value -= data.minTransferTime(stop);
            }
            return value;
        };
        auto adjustedDeparture = [&](const StopId stop, const std::uint32_t time) -> int64_t {
            int64_t value = static_cast<int64_t>(time);
            if (data.usesImplicitDepartureBufferTimes()) {
                value += data.minTransferTime(stop);
            }
            return value;
        };

        // Check forward translation layers and per-route stop sequences
        for (size_t rIdx = 0; rIdx < numRoutes; ++rIdx) {
            const RouteId flatRoute(rIdx);
            const PersistentRouteId pRoute = flatToPersistentRoute[rIdx];
            if (!pRoute.isValid() || static_cast<size_t>(pRoute) >= data.routes().size()) {
                error_msg << "Invalid persistent route id " << pRoute << " for flat route " << rIdx;
                return {false, error_msg.str()};
            }
            if (persistentToFlatRoute[static_cast<size_t>(pRoute)] != flatRoute) {
                error_msg << "Route translation mismatch for flat route " << rIdx;
                return {false, error_msg.str()};
            }

            const auto& pRouteObj = data.routes()[pRoute];
            if (pRouteObj.trips.empty()) {
                error_msg << "Flat route " << rIdx << " maps to an inactive persistent route " << pRoute;
                return {false, error_msg.str()};
            }

            const size_t stopSeqStart = qd.firstStopIdOfRoute[rIdx];
            const size_t stopSeqEnd = qd.firstStopIdOfRoute[rIdx + 1];
            const size_t numStopsOnRoute = stopSeqEnd - stopSeqStart;
            if (pRouteObj.stopSequence.size() != numStopsOnRoute) {
                error_msg << "Stop sequence length mismatch for route " << rIdx << ": queryData has "
                          << numStopsOnRoute << ", persistent has " << pRouteObj.stopSequence.size();
                return {false, error_msg.str()};
            }
            for (size_t s = 0; s < numStopsOnRoute; ++s) {
                if (qd.routeStopSequences[stopSeqStart + s] != pRouteObj.stopSequence[s]) {
                    error_msg << "Stop sequence mismatch for route " << rIdx << " at index " << s;
                    return {false, error_msg.str()};
                }
            }

            const size_t tripsInRoute = static_cast<size_t>(qd.firstTripOfRoute[rIdx + 1]) -
                                        static_cast<size_t>(qd.firstTripOfRoute[rIdx]);
            if (qd.routeLabels[rIdx].numberOfTrips != tripsInRoute) {
                error_msg << "RouteLabel trip count mismatch for route " << rIdx << ": label has "
                          << qd.routeLabels[rIdx].numberOfTrips << ", expected " << tripsInRoute;
                return {false, error_msg.str()};
            }
            const size_t expectedDepTimes = (numStopsOnRoute > 0 ? (numStopsOnRoute - 1) * tripsInRoute : 0);
            if (qd.routeLabels[rIdx].departureTimes.size() != expectedDepTimes) {
                error_msg << "RouteLabel departureTimes size mismatch for route " << rIdx << ": have "
                          << qd.routeLabels[rIdx].departureTimes.size() << ", expected " << expectedDepTimes;
                return {false, error_msg.str()};
            }
        }

        // Check forward translation layers and event integrity
        for (size_t tIdx = 0; tIdx < numTrips; ++tIdx) {
            const TripId flatTrip(tIdx);
            const PersistentTripId pTrip = flatToPersistentTrip[tIdx];
            if (!pTrip.isValid() || static_cast<size_t>(pTrip) >= data.trips().size()) {
                error_msg << "Invalid persistent trip id " << pTrip << " for flat trip " << tIdx;
                return {false, error_msg.str()};
            }
            if (persistentToFlatTrip[static_cast<size_t>(pTrip)] != flatTrip) {
                error_msg << "Trip translation mismatch for flat trip " << tIdx;
                return {false, error_msg.str()};
            }

            const auto& pTripObj = data.trips()[pTrip];
            if (!pTripObj.isActive) {
                error_msg << "Flat trip " << tIdx << " maps to an inactive persistent trip " << pTrip;
                return {false, error_msg.str()};
            }

            const PersistentRouteId pRoute = pTripObj.route;
            if (!pRoute.isValid() || static_cast<size_t>(pRoute) >= persistentToFlatRoute.size()) {
                error_msg << "Trip " << tIdx << " refers to invalid persistent route " << pRoute;
                return {false, error_msg.str()};
            }
            const RouteId expectedFlatRoute = persistentToFlatRoute[static_cast<size_t>(pRoute)];
            if (!expectedFlatRoute.isValid()) {
                error_msg << "Persistent route " << pRoute << " for trip " << tIdx << " has no flat mapping";
                return {false, error_msg.str()};
            }
            if (qd.routeOfTrip[tIdx] != expectedFlatRoute) {
                error_msg << "routeOfTrip mismatch for flat trip " << tIdx << ": expected " << expectedFlatRoute
                          << ", got " << qd.routeOfTrip[tIdx];
                return {false, error_msg.str()};
            }

            const size_t routeIdx = static_cast<size_t>(qd.routeOfTrip[tIdx]);
            const size_t stopSeqStart = qd.firstStopIdOfRoute[routeIdx];
            const size_t stopSeqEnd = qd.firstStopIdOfRoute[routeIdx + 1];
            const size_t numStopsOnRoute = stopSeqEnd - stopSeqStart;

            const size_t tripStartEvent = static_cast<size_t>(qd.firstStopEventOfTrip[tIdx]);
            const size_t tripEndEvent = static_cast<size_t>(qd.firstStopEventOfTrip[tIdx + 1]);

            if (tripEndEvent > numEvents) {
                error_msg << "Trip " << tIdx << " stop-event range exceeds number of events";
                return {false, error_msg.str()};
            }

            size_t nonSkipped = 0;
            for (std::uint32_t i = 0; i < pTripObj.numberOfEvents; ++i) {
                const auto& pEvent = data.events()[pTripObj.firstEvent + i];
                if (!pEvent.isSkipped) ++nonSkipped;
            }
            if (nonSkipped != numStopsOnRoute) {
                error_msg << "Non-skipped event count mismatch for trip " << tIdx << ": persistent has " << nonSkipped
                          << ", expected " << numStopsOnRoute;
                return {false, error_msg.str()};
            }

            for (size_t s = 0; s < numStopsOnRoute; ++s) {
                const size_t eventIndex = tripStartEvent + s;
                const StopId stop = qd.routeStopSequences[stopSeqStart + s];

                if (qd.eventLookup[eventIndex].stop != stop) {
                    error_msg << "EventLookup stop mismatch for trip " << tIdx << " at stop index " << s;
                    return {false, error_msg.str()};
                }
                if (qd.eventLookup[eventIndex].arrTime != qd.eventArrTimes[eventIndex]) {
                    error_msg << "EventLookup arrival time mismatch for stop event " << eventIndex;
                    return {false, error_msg.str()};
                }

                const size_t routeTripBegin = static_cast<size_t>(qd.firstTripOfRoute[routeIdx]);
                const size_t routeTripEnd = static_cast<size_t>(qd.firstTripOfRoute[routeIdx + 1]);
                const size_t tripsInRoute = routeTripEnd - routeTripBegin;
                const size_t tripOffset = tIdx - routeTripBegin;
                const size_t labelIdx = (s * tripsInRoute) + tripOffset;
                if (s + 1 < numStopsOnRoute && labelIdx < qd.routeLabels[routeIdx].departureTimes.size()) {
                    if (qd.routeLabels[routeIdx].departureTimes[labelIdx] != (int) qd.eventDepTimes[eventIndex]) {
                        error_msg << "RouteLabel departure time mismatch for trip " << tIdx << " at stop index " << s;
                        return {false, error_msg.str()};
                    }
                }
            }
        }

        for (size_t eIdx = 0; eIdx < numEvents; ++eIdx) {
            const PersistentStopEventId pEvent = flatToPersistentEvent[eIdx];
            if (!pEvent.isValid() || static_cast<size_t>(pEvent) >= data.events().size()) {
                error_msg << "Invalid persistent stop-event id " << pEvent << " for flat event " << eIdx;
                return {false, error_msg.str()};
            }
            if (persistentToFlatEvent[static_cast<size_t>(pEvent)] != StopEventId(eIdx)) {
                error_msg << "Stop-event translation mismatch for flat event " << eIdx;
                return {false, error_msg.str()};
            }
            if (data.events()[pEvent].isSkipped) {
                error_msg << "Flat event " << eIdx << " maps to a skipped persistent event " << pEvent;
                return {false, error_msg.str()};
            }
            const PersistentTripId pTrip = data.getTripOfEvent(pEvent);
            if (!data.trips()[pTrip].isActive) {
                error_msg << "Flat event " << eIdx << " maps to an inactive persistent trip " << pTrip;
                return {false, error_msg.str()};
            }
        }

        // Check reverse translation layers
        for (size_t pRouteIdx = 0; pRouteIdx < data.routes().size(); ++pRouteIdx) {
            const auto& pRoute = data.routes()[PersistentRouteId(pRouteIdx)];
            const RouteId flatRoute = persistentToFlatRoute[pRouteIdx];
            if (pRoute.trips.empty()) {
                if (flatRoute.isValid()) {
                    error_msg << "Inactive persistent route " << pRouteIdx << " maps to flat route " << flatRoute;
                    return {false, error_msg.str()};
                }
                continue;
            }
            if (!flatRoute.isValid() || static_cast<size_t>(flatRoute) >= flatToPersistentRoute.size()) {
                error_msg << "Persistent route " << pRouteIdx << " has invalid flat mapping";
                return {false, error_msg.str()};
            }
            if (flatToPersistentRoute[static_cast<size_t>(flatRoute)] != PersistentRouteId(pRouteIdx)) {
                error_msg << "Reverse route translation mismatch for persistent route " << pRouteIdx;
                return {false, error_msg.str()};
            }
        }

        for (size_t pTripIdx = 0; pTripIdx < data.trips().size(); ++pTripIdx) {
            const auto& pTrip = data.trips()[PersistentTripId(pTripIdx)];
            const TripId flatTrip = persistentToFlatTrip[pTripIdx];
            if (!pTrip.isActive) {
                if (flatTrip.isValid()) {
                    error_msg << "Inactive persistent trip " << pTripIdx << " maps to flat trip " << flatTrip;
                    return {false, error_msg.str()};
                }
                continue;
            }
            if (!flatTrip.isValid() || static_cast<size_t>(flatTrip) >= flatToPersistentTrip.size()) {
                error_msg << "Persistent trip " << pTripIdx << " has invalid flat mapping";
                return {false, error_msg.str()};
            }
            if (flatToPersistentTrip[static_cast<size_t>(flatTrip)] != PersistentTripId(pTripIdx)) {
                error_msg << "Reverse trip translation mismatch for persistent trip " << pTripIdx;
                return {false, error_msg.str()};
            }
        }

        for (size_t pEventIdx = 0; pEventIdx < data.events().size(); ++pEventIdx) {
            const auto& pEvent = data.events()[PersistentStopEventId(pEventIdx)];
            const StopEventId flatEvent = persistentToFlatEvent[pEventIdx];
            const PersistentTripId pTrip = data.getTripOfEvent(PersistentStopEventId(pEventIdx));
            const bool tripActive = data.trips()[pTrip].isActive;
            if (pEvent.isSkipped || !tripActive) {
                if (flatEvent.isValid()) {
                    error_msg << "Inactive or skipped persistent event " << pEventIdx << " maps to flat event "
                              << flatEvent;
                    return {false, error_msg.str()};
                }
                continue;
            }
            if (!flatEvent.isValid() || static_cast<size_t>(flatEvent) >= flatToPersistentEvent.size()) {
                error_msg << "Persistent event " << pEventIdx << " has invalid flat mapping";
                return {false, error_msg.str()};
            }
            if (flatToPersistentEvent[static_cast<size_t>(flatEvent)] != PersistentStopEventId(pEventIdx)) {
                error_msg << "Reverse event translation mismatch for persistent event " << pEventIdx;
                return {false, error_msg.str()};
            }
        }

        // Check per-trip time consistency (with implicit buffer times reverted)
        for (size_t tIdx = 0; tIdx < numTrips; ++tIdx) {
            const size_t routeIdx = static_cast<size_t>(qd.routeOfTrip[tIdx]);
            if (routeIdx >= numRoutes) {
                error_msg << "Trip " << tIdx << " references invalid route " << qd.routeOfTrip[tIdx];
                return {false, error_msg.str()};
            }
            const size_t stopSeqStart = qd.firstStopIdOfRoute[routeIdx];
            const size_t stopSeqEnd = qd.firstStopIdOfRoute[routeIdx + 1];
            const size_t numStopsOnRoute = stopSeqEnd - stopSeqStart;
            if (numStopsOnRoute == 0) continue;

            const size_t tripStartEvent = static_cast<size_t>(qd.firstStopEventOfTrip[tIdx]);
            const size_t tripEndEvent = static_cast<size_t>(qd.firstStopEventOfTrip[tIdx + 1]);
            if (tripEndEvent - tripStartEvent != numStopsOnRoute) {
                error_msg << "Trip " << tIdx << " stop-event count mismatch: expected " << numStopsOnRoute
                          << ", got " << (tripEndEvent - tripStartEvent);
                return {false, error_msg.str()};
            }

            StopId prevStop = qd.routeStopSequences[stopSeqStart];
            int64_t prevArr = adjustedArrival(prevStop, qd.eventArrTimes[tripStartEvent]);
            int64_t prevDep = adjustedDeparture(prevStop, qd.eventDepTimes[tripStartEvent]);
            if (prevDep < prevArr) {
                error_msg << "Departure before arrival on trip " << tIdx << " at stop index 0: arrival=" << prevArr
                          << ", departure=" << prevDep;
                return {false, error_msg.str()};
            }

            for (size_t s = 1; s < numStopsOnRoute; ++s) {
                const size_t eventIndex = tripStartEvent + s;
                const StopId stop = qd.routeStopSequences[stopSeqStart + s];
                const int64_t arr = adjustedArrival(stop, qd.eventArrTimes[eventIndex]);
                const int64_t dep = adjustedDeparture(stop, qd.eventDepTimes[eventIndex]);

                if (arr < prevArr) {
                    error_msg << "Decreasing arrival time on trip " << tIdx << " between stops " << (s - 1) << " and "
                              << s << ": " << prevArr << " -> " << arr;
                    return {false, error_msg.str()};
                }
                if (dep < prevDep) {
                    error_msg << "Decreasing departure time on trip " << tIdx << " between stops " << (s - 1) << " and "
                              << s << ": " << prevDep << " -> " << dep;
                    return {false, error_msg.str()};
                }
                if (dep < arr) {
                    error_msg << "Departure before arrival on trip " << tIdx << " at stop index " << s
                              << ": arrival=" << arr << ", departure=" << dep;
                    return {false, error_msg.str()};
                }

                prevArr = arr;
                prevDep = dep;
            }
        }

        // FIFO check per route (with implicit buffer times reverted)
        for (size_t rIdx = 0; rIdx < numRoutes; ++rIdx) {
            const size_t tripBegin = static_cast<size_t>(qd.firstTripOfRoute[rIdx]);
            const size_t tripEnd = static_cast<size_t>(qd.firstTripOfRoute[rIdx + 1]);
            if (tripEnd <= tripBegin + 1) continue;

            const size_t stopSeqStart = qd.firstStopIdOfRoute[rIdx];
            const size_t stopSeqEnd = qd.firstStopIdOfRoute[rIdx + 1];
            const size_t numStopsOnRoute = stopSeqEnd - stopSeqStart;
            if (numStopsOnRoute == 0) continue;

            for (size_t i = tripBegin; i + 1 < tripEnd; ++i) {
                const size_t tripA = i;
                const size_t tripB = i + 1;
                const size_t startA = static_cast<size_t>(qd.firstStopEventOfTrip[tripA]);
                const size_t startB = static_cast<size_t>(qd.firstStopEventOfTrip[tripB]);

                const StopId stop0 = qd.routeStopSequences[stopSeqStart];
                const int64_t depA0 = adjustedDeparture(stop0, qd.eventDepTimes[startA]);
                const int64_t depB0 = adjustedDeparture(stop0, qd.eventDepTimes[startB]);
                if (depA0 > depB0) {
                    error_msg << "FIFO violation on route " << rIdx << ": trip " << tripA << " departs at " << depA0
                              << " but next trip " << tripB << " departs earlier at " << depB0;
                    return {false, error_msg.str()};
                }

                for (size_t s = 0; s < numStopsOnRoute; ++s) {
                    const StopId stop = qd.routeStopSequences[stopSeqStart + s];
                    const size_t eventA = startA + s;
                    const size_t eventB = startB + s;
                    const int64_t arrA = adjustedArrival(stop, qd.eventArrTimes[eventA]);
                    const int64_t arrB = adjustedArrival(stop, qd.eventArrTimes[eventB]);
                    const int64_t depA = adjustedDeparture(stop, qd.eventDepTimes[eventA]);
                    const int64_t depB = adjustedDeparture(stop, qd.eventDepTimes[eventB]);

                    if (arrA > arrB) {
                        error_msg << "FIFO violation (arrival) on route " << rIdx << " at stop " << s << ": trip "
                                  << tripA << " arrives at " << arrA << " but next trip " << tripB
                                  << " arrives earlier at " << arrB;
                        return {false, error_msg.str()};
                    }
                    if (depA > depB) {
                        error_msg << "FIFO violation (departure) on route " << rIdx << " at stop " << s << ": trip "
                                  << tripA << " departs at " << depA << " but next trip " << tripB
                                  << " departs earlier at " << depB;
                        return {false, error_msg.str()};
                    }
                }
            }
        }

        // Check routeSegments consistency
        for (size_t stop = 0; stop < numStops; ++stop) {
            const size_t segBegin = qd.firstRouteSegmentOfStop[stop];
            const size_t segEnd = qd.firstRouteSegmentOfStop[stop + 1];
            for (size_t i = segBegin; i < segEnd; ++i) {
                const auto& segment = qd.routeSegments[i];
                const size_t routeIdx = static_cast<size_t>(segment.routeId);
                if (routeIdx >= numRoutes) {
                    error_msg << "RouteSegment has invalid route " << segment.routeId << " for stop " << stop;
                    return {false, error_msg.str()};
                }
                const size_t stopSeqStart = qd.firstStopIdOfRoute[routeIdx];
                const size_t stopSeqEnd = qd.firstStopIdOfRoute[routeIdx + 1];
                const size_t numStopsOnRoute = stopSeqEnd - stopSeqStart;
                if (static_cast<size_t>(segment.stopIndex) >= numStopsOnRoute) {
                    error_msg << "RouteSegment has invalid stop index " << segment.stopIndex << " on route "
                              << segment.routeId;
                    return {false, error_msg.str()};
                }
                if (qd.routeStopSequences[stopSeqStart + static_cast<size_t>(segment.stopIndex)] != StopId(stop)) {
                    error_msg << "RouteSegment mismatch: route " << segment.routeId << " stop index "
                              << segment.stopIndex << " does not match stop " << stop;
                    return {false, error_msg.str()};
                }
            }
        }

        return {true, "DynamicQueryData is valid"};
    }

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
        builder.firstStopEventOfTrip.resize(activeTripCount + 1);
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
        builder.firstStopEventOfTrip[activeTripCount] = StopEventId(activeEventCount);

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
