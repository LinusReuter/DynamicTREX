#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "../../../Helpers/Types.h"
#include "../../DynamicTimeTable/BuildQueryData.h"

namespace DynamicTB::Preprocessing {

/**
 * @brief Scratch buffers for transfer discovery, owned by the calling thread.
 */
struct DiscoveryWorkspace {
    /// Source segment considered by incoming discovery.
    struct SourceSegment {
        RouteId route;
        StopIndex i;
        Time footPathTime;
    };

    std::vector<PersistentStopEventId> desired;
    std::vector<std::pair<StopId, Time>> connectedStops;
    std::vector<SourceSegment> sources;
};

/**
 * @brief Pure timetable queries: given the current flat timetable, which transfers exist?
 *
 * This layer never touches the transfer store -- it only reads DynamicQueryData.
 */
class TransferDiscovery {
public:
    using DynamicQueryData = DynamicTimeTable::Algo::DynamicQueryData;

    explicit TransferDiscovery(const DynamicQueryData& queryData) noexcept : queryData_(&queryData) {}

    inline void setQueryData(const DynamicQueryData& queryData) noexcept { queryData_ = &queryData; }

    [[nodiscard]] inline const DynamicQueryData& queryData() const noexcept { return *queryData_; }

    /**
     * @brief Compute all feasible outgoing transfers from a single stop event.
     * Output is sorted ascending by PersistentStopEventId (required by the store diff).
     */
    inline void computeOutgoingTransfers(StopEventId flatFromEvent, DiscoveryWorkspace& ws,
                                         std::vector<PersistentStopEventId>& out) const {
        Time arrTime = queryData_->arrivalTimeOfEvent(flatFromEvent);
        if (arrTime == noTime) return;

        StopId fromStop = queryData_->stopOfEvent(flatFromEvent);
        const auto& qd = queryData_->queryData;
        TripId flatFromTrip = qd.tripOfStopEvent[flatFromEvent];
        RouteId fromRoute = qd.routeOfTrip[flatFromTrip];
        StopIndex fromIndex = queryData_->stopIndexOfEvent(flatFromEvent);

        if (fromIndex == StopIndex(0)) return;

        ws.connectedStops.clear();
        appendConnectedStops(fromStop, ws.connectedStops);

        for (const auto& [q, footPathTime] : ws.connectedStops) {
            Time minArr = arrTime + footPathTime;
            for (const auto& segment : qd.routesContainingStop(q)) {
                std::optional<TripId> optToTrip = findEarliestTripOnRoute(segment.routeId, segment.stopIndex, minArr);
                if (!optToTrip) continue;

                TripId toTrip = *optToTrip;
                if (segment.routeId == fromRoute && toTrip >= flatFromTrip && segment.stopIndex >= fromIndex) continue;
                if (isUTurn(flatFromTrip, fromIndex, toTrip, segment.stopIndex)) continue;

                out.push_back(queryData_->persistentEventId(toTrip, segment.stopIndex));
            }
        }
        std::ranges::sort(out);
    }

    /**
     * @brief Compute all feasible incoming transfers to a single stop event.
     * Output is sorted ascending by PersistentStopEventId (required by the store diff).
     */
    inline void computeIncomingTransfers(StopEventId flatToEvent, DiscoveryWorkspace& ws,
                                         std::vector<PersistentStopEventId>& out) const {
        Time toDepTime = queryData_->departureTimeOfEvent(flatToEvent);
        if (toDepTime == noTime) return;

        const auto& qd = queryData_->queryData;
        TripId toTrip = qd.tripOfStopEvent[flatToEvent];
        // Can't transfer to the last stop of a trip
        if (flatToEvent == (qd.firstStopEventOfTrip[toTrip + 1] - 1)) return;

        RouteId toRoute = qd.routeOfTrip[toTrip];
        StopIndex toIndex = queryData_->stopIndexOfEvent(flatToEvent);
        StopId toStop = queryData_->stopOfEvent(flatToEvent);

        // Profile optimization: find departure time of the immediate previous trip on the target route.
        // If a source trip can reach the previous trip, it shouldn't transfer to this one.
        TripId firstTripOfToRoute = qd.firstTripOfRoute[toRoute];
        Time prevDepTime = noTime;

        if (toTrip > firstTripOfToRoute) {
            prevDepTime = Time(qd.eventDepTimes[qd.firstStopEventOfTrip[toTrip - 1] + toIndex]);
        }

        ws.connectedStops.clear();
        ws.connectedStops.emplace_back(toStop, Time(0));
        const auto& rtg = qd.reverseTransferGraph;
        for (const auto edge : rtg.edgesFrom(toStop)) {
            ws.connectedStops.emplace_back(StopId(rtg.get(ToVertex, edge)), Time(rtg.get(TravelTime, edge)));
        }

        // Collect all potential source route segments
        auto& sources = ws.sources;
        sources.clear();
        sources.reserve(ws.connectedStops.size() * 4);  // Heuristic allocation

        for (const auto& [q, footPathTime] : ws.connectedStops) {
            for (const auto& segment : qd.routesContainingStop(q)) {
                if (segment.stopIndex == StopIndex(0)) continue;  // Can't transfer from the first stop of a trip
                sources.push_back({segment.routeId, segment.stopIndex, footPathTime});
            }
        }

        std::ranges::sort(sources, [](const DiscoveryWorkspace::SourceSegment& a,
                                      const DiscoveryWorkspace::SourceSegment& b) {
            if (a.route != b.route) return a.route < b.route;
            if (a.i != b.i) return a.i < b.i;
            return a.footPathTime < b.footPathTime;
        });

        // Scan sources and collect valid connections
        for (const auto& src : sources) {
            TripId firstTrip = qd.firstTripOfRoute[src.route];
            uint32_t numTrips = qd.firstTripOfRoute[src.route + 1] - firstTrip;
            if (numTrips == 0) continue;

            // Arrival time of a trip at this source's stop index, as int64 for window arithmetic.
            auto arrivalAt = [&](const TripId t) {
                return static_cast<int64_t>(queryData_->arrivalTimeOfEvent(queryData_->stopEventIdOfTripStop(t, src.i)));
            };

            // Leverage Consistency Invariant: if the first trip is noTime, exiting is forbidden for the entire route
            if (Time(qd.eventArrTimes[queryData_->stopEventIdOfTripStop(firstTrip, src.i)]) == noTime) continue;

            int64_t maxArr = static_cast<int64_t>(toDepTime) - static_cast<int64_t>(src.footPathTime);
            int64_t minArr = (prevDepTime != noTime)
                                 ? static_cast<int64_t>(prevDepTime) - static_cast<int64_t>(src.footPathTime)
                                 : -1;

            // Binary search across trips to locate the first candidate where arrTime > minArr
            int left = 0;
            int right = static_cast<int>(numTrips) - 1;
            int firstIdx = numTrips;

            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (arrivalAt(TripId(firstTrip + mid)) > minArr) {
                    firstIdx = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

            // Iterate forward to collect everything within the valid arrival window
            for (uint32_t idx = static_cast<uint32_t>(firstIdx); idx < numTrips; ++idx) {
                TripId t = TripId(firstTrip + idx);
                if (arrivalAt(t) > maxArr) break;  // Window closed; later trips will arrive too late

                // Filter out U-Turns and same-route forward invalidities
                if (src.route == toRoute && toTrip >= t && toIndex >= src.i) continue;
                if (isUTurn(t, src.i, toTrip, toIndex)) continue;

                PersistentStopEventId pEv =
                    queryData_->flatToPersistentEvent[queryData_->stopEventIdOfTripStop(t, src.i)];
                if (pEv.isValid()) out.push_back(pEv);
            }
        }
        std::ranges::sort(out);
    }

    /**
     * @brief Expand a stop into itself + footpath neighbors with transfer time.
     */
    inline void appendConnectedStops(StopId fromStop, std::vector<std::pair<StopId, Time>>& out) const {
        out.emplace_back(fromStop, 0);
        const auto& tg = queryData_->queryData.transferGraph;
        for (const auto edge : tg.edgesFrom(fromStop)) {
            auto toStop = StopId(tg.get(ToVertex, edge));
            auto travelTime = Time(tg.get(TravelTime, edge));
            out.emplace_back(toStop, travelTime);
        }
    }

    /**
     * @brief Filtering rule preventing temporal/topological U-turns.
     */
    [[nodiscard]] inline bool isUTurn(const TripId fromTrip, const StopIndex fromIndex, const TripId toTrip,
                                      const StopIndex toIndex) const noexcept {
        return TripBased::isUTurn(fromTrip, fromIndex, toTrip, toIndex, queryData_->queryData);
    }

    /**
     * @brief Finds the earliest available trip matching arrival constraints via binary search.
     */
    [[nodiscard]] inline std::optional<TripId> findEarliestTripOnRoute(const RouteId route, const StopIndex stopIndex,
                                                                      const Time minDepartureTime) const {
        const auto& routeLabel = queryData_->queryData.routeLabels[route];
        const uint32_t numTrips = routeLabel.numberOfTrips;

        if (numTrips == 0) return std::nullopt;

        const size_t stopSeqStart = queryData_->queryData.firstStopIdOfRoute[route];
        const size_t stopSeqEnd = queryData_->queryData.firstStopIdOfRoute[route + 1];
        const size_t numStops = stopSeqEnd - stopSeqStart;
        if (numStops < 2) return std::nullopt;
        if (static_cast<size_t>(stopIndex) + 1 >= numStops) return std::nullopt;  // no departure at last stop

        const size_t baseOffset = static_cast<size_t>(stopIndex) * numTrips;
        if (baseOffset >= routeLabel.departureTimes.size()) return std::nullopt;

        int left = 0;
        int right = static_cast<int>(numTrips) - 1;
        int bestTrip = -1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (Time(routeLabel.departureTimes[baseOffset + static_cast<size_t>(mid)]) >= minDepartureTime) {
                bestTrip = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        if (bestTrip != -1) {
            return TripId(queryData_->queryData.firstTripOfRoute[route] + bestTrip);
        }
        return std::nullopt;
    }

private:
    const DynamicQueryData* queryData_{nullptr};
};

}  // namespace DynamicTB::Preprocessing
