#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../Algorithms/UnionFind.h"
#include "../../Helpers/IO/Serialization.h"
#include "../../Helpers/Types.h"
#include "../RAPTOR/Data.h"
#include "Entities/PersistentRoute.h"
#include "Entities/PersistentStopEvent.h"
#include "Entities/PersistentTrip.h"
#include "UpdateTypes.h"

namespace DynamicTimeTable {
namespace Algo {
    struct UpdatePipeline;
}
}

namespace DynamicTimeTable {

/**
 * DynamicTimeTable
 *
 * Persistent ids are dense and directly index the corresponding vectors.
 * Trips own a contiguous stop-event block.
 * Routes have an immutable stop sequence; if a trip's effective sequence changes, it migrates.
 */
class Data {
    friend struct ::DynamicTimeTable::Algo::UpdatePipeline;

public:
    Data() = default;

    explicit Data(const std::string& fileName) { deserialize(fileName); }

    explicit Data(const RAPTOR::Data& raptorData) { importFromRaptor(raptorData); }

    void importFromRaptor(const RAPTOR::Data& raptorData) {
        transferGraph_ = raptorData.transferGraph;
        numberOfStops_ = raptorData.numberOfStops();

        minTransferTimes_.assign(numberOfStops_, 0);
        for (StopId stop(0); stop < StopId(numberOfStops_); ++stop) {
            minTransferTimes_[static_cast<size_t>(stop)] = raptorData.stopData[stop].minTransferTime;
        }

        // Some RAPTOR accessors are non-const in this codebase, keep behavior consistent.
        RAPTOR::Data& rd = const_cast<RAPTOR::Data&>(raptorData);

        // Pre-allocate based on RAPTOR data sizes to limit reallocations
        const std::size_t numRoutes = rd.numberOfRoutes();
        const std::size_t totalTrips = rd.numberOfTrips();
        const std::size_t totalEvents = rd.numberOfStopEvents();

        routes_.clear();
        routes_.reserve(numRoutes);

        trips_.clear();
        trips_.reserve(totalTrips);

        events_.clear();
        events_.reserve(totalEvents);

        eventToTrip_.clear();
        eventToTrip_.reserve(totalEvents);

        // Ensure implicit buffer times from RAPTOR are applied so they are
        // imported directly as the times of the dynamic timetable
        rd.useImplicitDepartureBufferTimes();

        implicitDepartureBufferTimes_ = rd.implicitDepartureBufferTimes;
        implicitArrivalBufferTimes_ = rd.implicitArrivalBufferTimes;

        for (std::size_t r = 0; r < numRoutes; r++) {
            const RouteId staticRouteId(r);

            PersistentRoute route;
            route.routeId = PersistentRouteId(routes_.size());

            const auto stopsOfRoute = rd.stopsOfRoute(staticRouteId);
            route.stopSequence.reserve(stopsOfRoute.size());
            for (const StopId stop : stopsOfRoute) {
                route.stopSequence.push_back(stop);
            }

            const std::size_t numTrips = rd.numberOfTripsInRoute(staticRouteId);
            const std::size_t numStops = stopsOfRoute.size();
            const auto stopEventsOfRoute = rd.stopEventsOfRoute(staticRouteId);

            for (std::size_t tripIdx = 0; tripIdx < numTrips; tripIdx++) {
                const PersistentTripId tripId(trips_.size());

                PersistentTrip trip;
                trip.route = route.routeId;
                trip.isActive = true;
                trip.firstEvent = PersistentStopEventId(events_.size());
                trip.numberOfEvents = static_cast<std::uint32_t>(numStops);

                for (std::size_t stopIdx = 0; stopIdx < numStops; stopIdx++) {
                    const RAPTOR::StopEvent& e = stopEventsOfRoute[tripIdx * numStops + stopIdx];

                    PersistentStopEvent pe;
                    pe.stop = stopsOfRoute[stopIdx];
                    pe.arrivalTime = Time(e.arrivalTime);
                    pe.departureTime = Time(e.departureTime);
                    pe.isSkipped = false;

                    events_.push_back(pe);

                    eventToTrip_.push_back(tripId);
                }

                trips_.push_back(trip);
                route.trips.push_back(tripId);
            }

            routes_.push_back(route);
        }

        rebuildRoutesBySequenceHash();
    }

    // --- Change Summary ---

    const ChangeSummary& getLatestChanges() const { return latestChanges_; }

    void clearChangeSummary() { latestChanges_.clear(); }

    // --- Basic Getters ---

    std::size_t numberOfStops() const { return numberOfStops_; }

    std::size_t numberOfRoutes() const { return routes_.size(); }

    std::size_t numberOfActiveTrips() const {
        std::size_t count = 0;
        for (std::size_t i = 0; i < trips_.size(); i++) {
            if (trips_[PersistentTripId(i)].isActive) count++;
        }
        return count;
    }

    std::size_t numberOfMintedStopEvents() const { return events_.size(); }

    const PersistentRoute& getRoute(const PersistentRouteId routeId) const { return routes_[routeId]; }

    const std::vector<PersistentRoute>& routes() const noexcept { return routes_; }
    const std::vector<PersistentTrip>& trips() const noexcept { return trips_; }
    const std::vector<PersistentStopEvent>& events() const noexcept { return events_; }
    const TransferGraph& transferGraph() const noexcept { return transferGraph_; }

    const std::vector<int>& minTransferTimes() const noexcept { return minTransferTimes_; }

    int minTransferTime(const StopId stop) const noexcept { return minTransferTimes_[stop]; }

    bool usesImplicitDepartureBufferTimes() const noexcept { return implicitDepartureBufferTimes_; }

    bool usesImplicitArrivalBufferTimes() const noexcept { return implicitArrivalBufferTimes_; }

    // --- O(1) resolution helpers ---

    PersistentRouteId getRouteOfEvent(const PersistentStopEventId eventId) const {
        return trips_[eventToTrip_[eventId]].route;
    }

    PersistentTripId getTripOfEvent(const PersistentStopEventId eventId) const { return eventToTrip_[eventId]; }

    StopIndex getStopIndexOfEvent(const PersistentStopEventId eventId) const {
        return StopIndex(eventId - trips_[eventToTrip_[eventId]].firstEvent);
    }

    const PersistentTrip* getTrip(const PersistentTripId tripId) const {
        if (!isTrip(tripId)) return nullptr;
        return &trips_[tripId];
    }

    const PersistentStopEvent* getEvent(const PersistentStopEventId eventId) const {
        if (!isEvent(eventId)) return nullptr;
        return &events_[eventId];
    }

    std::optional<PersistentStopEventId> getEventId(const PersistentTripId tripId, const StopIndex stopIndex) const {
        if (!isTrip(tripId)) return std::nullopt;
        const PersistentTrip& trip = trips_[tripId];

        const std::size_t idx = static_cast<std::size_t>(stopIndex);
        if (idx >= trip.numberOfEvents) return std::nullopt;

        const std::size_t first = static_cast<std::size_t>(trip.firstEvent);
        return PersistentStopEventId(first + idx);
    }

    // --- Partitioning/Layout placeholders (kept for compatibility) ---

    void createCompactLayoutGraph() {}

    void applyGlobalIDs([[maybe_unused]] const std::vector<uint64_t>& globalIds) noexcept {}

    void readPartitionFile([[maybe_unused]] const std::string& fileName) {}

    // --- Serialization ---

    void serialize(const std::string& fileName) const noexcept {
        IO::serialize(fileName, routes_, trips_, events_, routesBySequenceHash_, eventToTrip_, transferGraph_,
                      numberOfStops_, cellIds_, unionFind_, layoutGraph_, latestChanges_, minTransferTimes_,
                      implicitDepartureBufferTimes_, implicitArrivalBufferTimes_);
    }

    void deserialize(const std::string& fileName) noexcept {
        IO::deserialize(fileName, routes_, trips_, events_, routesBySequenceHash_, eventToTrip_, transferGraph_,
                        numberOfStops_, cellIds_, unionFind_, layoutGraph_, latestChanges_, minTransferTimes_,
                        implicitDepartureBufferTimes_, implicitArrivalBufferTimes_);
    }

    void printInfo() const {
        std::cout << "DynamicTimeTable info\n";
        std::cout << "-----------------------------\n";
        std::cout << "Routes             : " << numberOfRoutes() << "\n";
        std::cout << "Active Trips       : " << numberOfActiveTrips() << "\n";
        std::cout << "Minted Stop Events : " << numberOfMintedStopEvents() << "\n";
    }

private:
    bool isRoute(const PersistentRouteId id) const { return static_cast<std::size_t>(id) < routes_.size(); }

    bool isTrip(const PersistentTripId id) const { return static_cast<std::size_t>(id) < trips_.size(); }

    bool isEvent(const PersistentStopEventId id) const { return static_cast<std::size_t>(id) < events_.size(); }

    std::size_t hashStopSequence(const std::vector<StopId>& sequence) const {
        std::size_t h = 0xcbf29ce484222325ULL;
        for (const StopId s : sequence) {
            const std::size_t x = static_cast<std::size_t>(static_cast<std::uint32_t>(s));
            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }

    bool stopSequenceEquals(const std::vector<StopId>& a, const std::vector<StopId>& b) const {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    void rebuildRoutesBySequenceHash() {
        routesBySequenceHash_.clear();
        routesBySequenceHash_.reserve(routes_.size());
        for (std::size_t i = 0; i < routes_.size(); i++) {
            const PersistentRouteId r(i);
            const std::size_t h = hashStopSequence(routes_[r].stopSequence);
            routesBySequenceHash_[h].push_back(r);
        }
    }

    // Persistent workspace storage

    std::vector<PersistentRoute> routes_;
    std::vector<PersistentTrip> trips_;
    std::vector<PersistentStopEvent> events_;

    std::unordered_map<std::size_t, std::vector<PersistentRouteId>> routesBySequenceHash_;

    // O(1) event-resolution helpers
    std::vector<PersistentTripId> eventToTrip_;

    // Static topology extracted from RAPTOR (currently used internally)
    TransferGraph transferGraph_;
    std::size_t numberOfStops_ = 0;
    std::vector<int> minTransferTimes_;
    bool implicitDepartureBufferTimes_ = false;
    bool implicitArrivalBufferTimes_ = false;

    // Partitioning/Layout (kept for compatibility with existing code paths)
    std::vector<uint16_t> cellIds_;
    UnionFind unionFind_;
    StaticGraphWithWeightsAndCoordinates layoutGraph_;

    ChangeSummary latestChanges_;
};

}  // namespace DynamicTimeTable
