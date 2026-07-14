/**********************************************************************************

 Copyright (c) 2023-2025 Patrick Steil

 MIT License

 Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**********************************************************************************/
#pragma once

#include <cassert>

#include "../../../Helpers/Types.h"
#include "../../../DataStructures/TripBased/Data.h"

namespace TripBased {
/*
struct EdgeLabel {
EdgeLabel(const StopEventId stopEvent = noStopEvent,
        const TripId trip = noTripId,
        const StopEventId firstEvent = noStopEvent)
  : stopEvent(stopEvent), trip(trip), firstEvent(firstEvent) {}

StopEventId stopEvent;
TripId trip;
StopEventId firstEvent;

StopEventId getStopEvent() const { return stopEvent; }
TripId getTrip() const { return trip; }
StopEventId getFirstEvent() const { return firstEvent; }

void setStopEvent(StopEventId id) { stopEvent = id; }
void setTrip(TripId id) { trip = id; }
void setFirstEvent(StopEventId id) { firstEvent = id; }
};
*/

struct EdgeLabel {
    uint64_t data;

    EdgeLabel(StopIndex stopIndex = noStopIndex, TripId trip = noTripId, StopEventId firstEvent = noStopEvent)
        : data(0) {
        setTrip(trip);
        setFirstEvent(firstEvent);
        setStopIndex(stopIndex);
    }

    void init(const StopEventId event, const TripId trip, const StopEventId firstEvent) {
        setTrip(trip);
        setFirstEvent(firstEvent);
        // TODO WHY +1 ????
        setStopIndex(StopIndex(event - firstEvent + 1));
    }

    StopIndex getStopIndex() const { return static_cast<StopIndex>(data & 0xFFULL); }
    void setStopIndex(StopIndex d) { data = (data & ~0xFFULL) | d; }

    TripId getTrip() const { return static_cast<TripId>((data >> 8) & 0x7FFFFFULL); }

    void setTrip(TripId id) {
        assert(id < (1u << 23) || id == noTripId);
        data = (data & ~(0x7FFFFFULL << 8)) | (static_cast<uint64_t>(id & 0x7FFFFF) << 8);
    }

    StopEventId getFirstEvent() const { return static_cast<StopEventId>((data >> 31) & 0x7FFFFFFULL); }

    void setFirstEvent(StopEventId id) {
        assert(id < (1u << 27) || id == noStopEvent);
        data = (data & ~(0x7FFFFFFULL << 31)) | (static_cast<uint64_t>(id & 0x7FFFFFFULL) << 31);
    }

    StopEventId getStopEvent() const { return StopEventId(getFirstEvent() + getStopIndex()); }

    uint8_t getRank() const { return static_cast<uint8_t>((data >> 58) & 0x1FULL); }

    void setRank(uint8_t value) {
        assert(value <= 16);
        data = (data & ~(0x1FULL << 58)) | (static_cast<uint64_t>(value & 0x1F) << 58);
    }
};

static_assert(sizeof(EdgeLabel) == 8, "EdgeLabel must be 8 bytes");
static_assert(alignof(EdgeLabel) == alignof(uint64_t), "Unexpected alignment");

struct EdgeLabelCellId {
    uint64_t data;
    uint16_t cellId;

    EdgeLabelCellId(StopIndex stopIndex = noStopIndex, TripId trip = noTripId, StopEventId firstEvent = noStopEvent,
                    uint16_t cell = 0)
        : data(0), cellId(cell) {
        setTrip(trip);
        setFirstEvent(firstEvent);
        setStopIndex(stopIndex);
    }

    StopIndex getStopIndex() const { return static_cast<StopIndex>(data & 0xFFULL); }
    void setStopIndex(StopIndex d) { data = (data & ~0xFFULL) | d; }

    TripId getTrip() const { return static_cast<TripId>((data >> 8) & 0x7FFFFFULL); }
    void setTrip(TripId id) {
        assert(id < (1u << 23) || id == noTripId);
        data = (data & ~(0x7FFFFFULL << 8)) | (static_cast<uint64_t>(id & 0x7FFFFF) << 8);
    }

    StopEventId getFirstEvent() const { return static_cast<StopEventId>((data >> 31) & 0x7FFFFFFULL); }
    void setFirstEvent(StopEventId id) {
        assert(id < (1u << 27) || id == noStopEvent);
        data = (data & ~(0x7FFFFFFULL << 31)) | (static_cast<uint64_t>(id & 0x7FFFFFFULL) << 31);
    }

    StopEventId getStopEvent() const { return StopEventId(getFirstEvent() + getStopIndex()); }

    uint8_t getRank() const { return static_cast<uint8_t>((data >> 58) & 0x1FULL); }

    void setRank(uint8_t value) {
        assert(value <= 16);
        data = (data & ~(0x1FULL << 58)) | (static_cast<uint64_t>(value & 0x1F) << 58);
    }

    uint16_t getCellId() const { return cellId; }
    void setCellId(uint16_t id) { cellId = id; }
};

struct RouteLabel {
    RouteLabel() : numberOfTrips(0) {}

    inline StopIndex end() const noexcept {
        return StopIndex(departureTimes.size() / numberOfTrips);
    }

    inline StopIndex getStopIndex(const size_t index) const noexcept {
        return StopIndex(static_cast<int>(index) / end());
    }

    inline size_t getTripOffset(const size_t index) const noexcept {
        return index % end();
    }

    u_int32_t numberOfTrips;
    std::vector<int> departureTimes;
};

struct EventLookup {
    StopId stop;
    uint32_t arrTime;

    EventLookup(const StopId stop = noStop, uint32_t arrTime = 0) : stop(stop), arrTime(arrTime) {}
};

struct QueryDataBuilder {
    TransferGraph transferGraph;
    TransferGraph reverseTransferGraph;

    std::vector<EventLookup> eventLookup;
    std::vector<std::uint32_t> eventArrTimes;
    std::vector<std::uint32_t> eventDepTimes;

    std::vector<TripId> tripOfStopEvent;
    std::vector<RouteId> routeOfTrip;
    std::vector<StopEventId> firstStopEventOfTrip;
    std::vector<TripId> firstTripOfRoute;

    std::vector<size_t> firstRouteSegmentOfStop;
    std::vector<RAPTOR::RouteSegment> routeSegments;
    std::vector<size_t> firstStopIdOfRoute;
    std::vector<StopId> routeStopSequences;
    std::vector<RouteLabel> routeLabels;
};

class QueryData {
public:
    QueryData(QueryDataBuilder&& builder)
        : transferGraph(std::move(builder.transferGraph)),
          reverseTransferGraph(std::move(builder.reverseTransferGraph)),
          eventLookup(std::move(builder.eventLookup)),
          eventArrTimes(std::move(builder.eventArrTimes)),
          eventDepTimes(std::move(builder.eventDepTimes)),
          tripOfStopEvent(std::move(builder.tripOfStopEvent)),
          routeOfTrip(std::move(builder.routeOfTrip)),
          firstStopEventOfTrip(std::move(builder.firstStopEventOfTrip)),
          firstTripOfRoute(std::move(builder.firstTripOfRoute)),
          firstRouteSegmentOfStop(std::move(builder.firstRouteSegmentOfStop)),
          routeSegments(std::move(builder.routeSegments)),
          firstStopIdOfRoute(std::move(builder.firstStopIdOfRoute)),
          routeStopSequences(std::move(builder.routeStopSequences)),
          routeLabels(std::move(builder.routeLabels)) {}

    QueryData(const Data& data)
        : transferGraph(data.raptorData.transferGraph),
          reverseTransferGraph(data.raptorData.transferGraph),
          eventLookup(data.numberOfStopEvents()),
          eventArrTimes(data.numberOfStopEvents()),
          eventDepTimes(data.numberOfStopEvents()),
          tripOfStopEvent(data.tripOfStopEvent),
          routeOfTrip(data.routeOfTrip),
          firstStopEventOfTrip(data.firstStopEventOfTrip),
          firstTripOfRoute(data.firstTripOfRoute),
          firstRouteSegmentOfStop(data.raptorData.firstRouteSegmentOfStop),
          routeSegments(data.raptorData.routeSegments),
          firstStopIdOfRoute(data.firstStopIdOfTrip),
          routeStopSequences(data.raptorData.stopIds),
          routeLabels(data.numberOfRoutes()) {
        reverseTransferGraph.revert();

#pragma omp parallel for
        for (size_t event = 0; event < data.numberOfStopEvents(); ++event) {
            eventLookup[event] = EventLookup(data.arrivalEvents[event].stop, data.arrivalEvents[event].arrivalTime);
            eventArrTimes[event] = data.arrivalEvents[event].arrivalTime;
            eventDepTimes[event] = data.raptorData.stopEvents[event].departureTime;
        }

        for (const RouteId route : data.raptorData.routes()) {
            const size_t numberOfStops = data.numberOfStopsInRoute(route);
            const size_t numberOfTrips = data.raptorData.numberOfTripsInRoute(route);
            const RAPTOR::StopEvent* stopEvents = data.raptorData.firstTripOfRoute(route);
            routeLabels[route].numberOfTrips = numberOfTrips;
            routeLabels[route].departureTimes.resize((numberOfStops - 1) * numberOfTrips);
            for (size_t trip = 0; trip < numberOfTrips; trip++) {
                for (size_t stopIndex = 0; stopIndex + 1 < numberOfStops; stopIndex++) {
                    routeLabels[route].departureTimes[(stopIndex * numberOfTrips) + trip] =
                        stopEvents[(trip * numberOfStops) + stopIndex].departureTime;
                }
            }
        }
    }

    inline RouteId getRouteOfStopEvent(const StopEventId stopEvent) const noexcept {
        return routeOfTrip[tripOfStopEvent[stopEvent]];
    }

    inline SubRange<std::vector<RAPTOR::RouteSegment>> routesContainingStop(const StopId stop) const noexcept {
        return SubRange<std::vector<RAPTOR::RouteSegment>>(routeSegments, firstRouteSegmentOfStop, stop);
    }

    inline const StopId* stopArrayOfRoute(const RouteId route) const noexcept {
        return &(routeStopSequences[firstStopIdOfRoute[route]]);
    }

public:
    TransferGraph transferGraph;
    TransferGraph reverseTransferGraph;

    std::vector<EventLookup> eventLookup;  // Stop and arrival time
    std::vector<std::uint32_t> eventArrTimes;
    std::vector<std::uint32_t> eventDepTimes;

    std::vector<TripId> tripOfStopEvent;
    std::vector<RouteId> routeOfTrip;
    std::vector<StopEventId> firstStopEventOfTrip;
    std::vector<TripId> firstTripOfRoute;

    std::vector<size_t> firstRouteSegmentOfStop;
    std::vector<RAPTOR::RouteSegment> routeSegments;
    std::vector<size_t> firstStopIdOfRoute;
    std::vector<StopId> routeStopSequences;
    // Departure times of events, sorted by (stopIndex,trip)
    std::vector<RouteLabel> routeLabels;
};

struct Transfers {
    Transfers(const Data& data)
        : beginOut(data.stopEventGraph.getBeginOut()),
          labels(data.stopEventGraph.numEdges()),
          travelTime(data.stopEventGraph.get(TravelTime)) {
        for (const Edge edge : data.stopEventGraph.edges()) {
            const StopEventId event(data.stopEventGraph.get(ToVertex, edge));
            const TripId trip = data.tripOfStopEvent[event];
            const StopEventId firstEvent = data.firstStopEventOfTrip[trip];
            labels[edge].init(event, trip, firstEvent);
        }
    }
    Transfers(std::vector<Edge> beginOut, std::vector<EdgeLabel> labels, std::vector<int> travelTime) noexcept
        : beginOut(std::move(beginOut)), labels(std::move(labels)), travelTime(std::move(travelTime)) {}

    std::vector<Edge> beginOut;
    std::vector<EdgeLabel> labels;
    std::vector<int> travelTime;
};

// A lightweight structure representing a directional topological edge
struct SimpleEdge {
    uint32_t from;
    uint32_t to;

    // Standard operators to allow sorting and unique comparison
    bool operator<(const SimpleEdge& fire) const {
        return std::tie(from, to) < std::tie(fire.from, fire.to);
    }

    bool operator==(const SimpleEdge& fire) const {
        return std::tie(from, to) == std::tie(fire.from, fire.to);
    }
};

// Holds the descriptive delta between the two sets
struct TransferComparisonResult {
    std::vector<SimpleEdge> onlyInFirst;
    std::vector<SimpleEdge> onlyInSecond;

    bool areEqual() const { return onlyInFirst.empty() && onlyInSecond.empty(); }
    bool isFirstSuperset() const { return !onlyInFirst.empty() && onlyInSecond.empty(); }
    bool isSecondSuperset() const { return onlyInFirst.empty() && !onlyInSecond.empty(); }
};

/**
 * Helper function to unpack the CSR (Compressed Sparse Row) representation
 * into a sorted list of unique topological edges.
 */
inline std::vector<SimpleEdge> extractTopology(const Transfers& transfers) {
    std::vector<SimpleEdge> edges;
    if (transfers.beginOut.empty()) return edges;

    // The number of source vertices is beginOut.size() - 1
    for (size_t fromVertex = 0; fromVertex < transfers.beginOut.size() - 1; ++fromVertex) {
        size_t edgeBegin = transfers.beginOut[fromVertex];
        size_t edgeEnd = transfers.beginOut[fromVertex + 1];

        for (size_t edgeIdx = edgeBegin; edgeIdx < edgeEnd; ++edgeIdx) {
            // Guard against potential out-of-bounds if building custom/malformed data
            if (edgeIdx < transfers.labels.size()) {
                // TODO Correction of + 1
                uint32_t toVertex = static_cast<uint32_t>(transfers.labels[edgeIdx].getStopEvent() -1);
                edges.push_back({static_cast<uint32_t>(fromVertex), toVertex});
            }
        }
    }
    return edges;
}

/**
 * Compares two Transfers instances purely by their network topology.
 * Returns lists of edges unique to either the first or second instance.
 *
 * Both instances are expected to share the same CSR vertex numbering (i.e. they were
 * exported for the same DynamicQueryData), so rows can be compared directly by index
 * instead of materializing and sorting the full edge lists of both graphs.
 */
inline TransferComparisonResult compareTransfers(const Transfers& lhs, const Transfers& rhs) {
    TransferComparisonResult result;
    if (lhs.beginOut.empty() || rhs.beginOut.empty()) return result;

    assert(lhs.beginOut.size() == rhs.beginOut.size());
    const std::size_t numVertices = lhs.beginOut.size() - 1;

    std::vector<uint32_t> lhsTargets;
    std::vector<uint32_t> rhsTargets;

    for (std::size_t fromVertex = 0; fromVertex < numVertices; ++fromVertex) {
        const std::size_t lhsBegin = lhs.beginOut[fromVertex];
        const std::size_t lhsEnd = lhs.beginOut[fromVertex + 1];
        const std::size_t rhsBegin = rhs.beginOut[fromVertex];
        const std::size_t rhsEnd = rhs.beginOut[fromVertex + 1];

        lhsTargets.clear();
        rhsTargets.clear();
        lhsTargets.reserve(lhsEnd - lhsBegin);
        rhsTargets.reserve(rhsEnd - rhsBegin);

        for (std::size_t edgeIdx = lhsBegin; edgeIdx < lhsEnd; ++edgeIdx) {
            lhsTargets.push_back(static_cast<uint32_t>(lhs.labels[edgeIdx].getStopEvent() - 1));
        }
        for (std::size_t edgeIdx = rhsBegin; edgeIdx < rhsEnd; ++edgeIdx) {
            rhsTargets.push_back(static_cast<uint32_t>(rhs.labels[edgeIdx].getStopEvent() - 1));
        }

        std::sort(lhsTargets.begin(), lhsTargets.end());
        std::sort(rhsTargets.begin(), rhsTargets.end());

        const uint32_t from = static_cast<uint32_t>(fromVertex);
        std::size_t li = 0;
        std::size_t ri = 0;
        while (li < lhsTargets.size() && ri < rhsTargets.size()) {
            if (lhsTargets[li] < rhsTargets[ri]) {
                result.onlyInFirst.push_back({from, lhsTargets[li]});
                ++li;
            } else if (rhsTargets[ri] < lhsTargets[li]) {
                result.onlyInSecond.push_back({from, rhsTargets[ri]});
                ++ri;
            } else {
                ++li;
                ++ri;
            }
        }
        while (li < lhsTargets.size()) {
            result.onlyInFirst.push_back({from, lhsTargets[li++]});
        }
        while (ri < rhsTargets.size()) {
            result.onlyInSecond.push_back({from, rhsTargets[ri++]});
        }
    }

    return result;
}


inline bool isUTurn(const TripId fromTrip, const StopIndex fromIndex, const TripId toTrip,
                    const StopIndex toIndex, const QueryData& qd) noexcept {
    if (fromIndex < 2) return false;
    auto num_stops_to = qd.firstStopEventOfTrip[toTrip + 1] - qd.firstStopEventOfTrip[toTrip];
    if (toIndex + 1 >= num_stops_to) return false;
    auto stop_sq_from = qd.stopArrayOfRoute(qd.routeOfTrip[fromTrip]);
    auto stop_sq_to = qd.stopArrayOfRoute(qd.routeOfTrip[toTrip]);
    if (stop_sq_from[fromIndex - 1] != stop_sq_to[toIndex + 1]) return false;
    if (qd.eventArrTimes[qd.firstStopEventOfTrip[fromTrip] + fromIndex - 1] >
        qd.eventDepTimes[qd.firstStopEventOfTrip[toTrip] + toIndex + 1])
        return false;
    return true;
}

inline bool isSameRouteForward(const TripId fromTrip, const StopIndex fromIndex, const TripId toTrip, const StopIndex toIndex,
                        const QueryData& qd) noexcept {
    return (qd.routeOfTrip[fromTrip] == qd.routeOfTrip[toTrip]) && (toTrip >= fromTrip) && (toIndex >= fromIndex);
}

inline bool isValidTransfer(const TripId fromTrip, const StopIndex fromIndex, const TripId toTrip, const StopIndex toIndex,
                            const QueryData& qd) noexcept {
    if (fromIndex == 0) return false;
    auto num_stops_to = qd.firstStopEventOfTrip[toTrip + 1] - qd.firstStopEventOfTrip[toTrip];
    if (toIndex >= num_stops_to) return false;
    if (isUTurn(fromTrip, fromIndex, toTrip, toIndex, qd)) return false;
    if (isSameRouteForward(fromTrip, fromIndex, toTrip, toIndex, qd)) return false;
    auto footpath_time = 0;
    auto from_route = qd.routeOfTrip[fromTrip];
    auto stop_from = qd.stopArrayOfRoute(from_route)[fromIndex];
    auto to_route = qd.routeOfTrip[toTrip];
    auto stop_to = qd.stopArrayOfRoute(to_route)[toIndex];
    if (stop_from != stop_to) {
        auto edge = qd.transferGraph.findEdge(stop_from, stop_to);
        if (edge == noEdge) {
            std::cout << "No edge found for " << stop_from << " -> " << stop_to << std::endl;
            return false;
        }
        footpath_time = qd.transferGraph.get(TravelTime, qd.transferGraph.findEdge(Vertex(stop_from.value()), Vertex(stop_to.value())));
    }
    auto arr_time = qd.eventArrTimes[qd.firstStopEventOfTrip[fromTrip] + fromIndex];
    auto dep_time = qd.eventDepTimes[qd.firstStopEventOfTrip[toTrip] + toIndex];
    if (arr_time + footpath_time > dep_time) return false;
    return true;
}

/**
 * Validate Transfers on basic invariants:
 * - No transfers from a trips first Stop
 * - No transfers to a trips last Stop
 * - arrival + footpath <= departure
 * - No U-Turns (same trip, later stop -> earlier stop)
 * - No same-route forward transfers (same route, later trip -> earlier trip)
 */
inline bool validateTransfers(const Transfers& transfers, const QueryData& qd) {
    bool ret = true;
    std::vector<SimpleEdge> edges = extractTopology(transfers);

    #pragma omp parallel for
    for (auto edge : edges) {
        auto from_stop_idx = edge.from - qd.firstStopEventOfTrip[qd.tripOfStopEvent[edge.from]];
        auto to_stop_idx = edge.to - qd.firstStopEventOfTrip[qd.tripOfStopEvent[edge.to]];
        bool valid = isValidTransfer(qd.tripOfStopEvent[edge.from], StopIndex(from_stop_idx), qd.tripOfStopEvent[edge.to], StopIndex(to_stop_idx), qd);
        if (!valid) {
            std::cout << "Invalid transfer from event " << edge.from << " (trip " << qd.tripOfStopEvent[edge.from] << ", stop index " << from_stop_idx << ") to event " << edge.to << " (trip " << qd.tripOfStopEvent[edge.to] << ", stop index " << to_stop_idx << ")\n";
            ret = false;
        }
    }
    return ret;
}

}  // namespace TripBased
