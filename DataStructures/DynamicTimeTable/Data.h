#pragma once

#include "../TripBased/Data.h"
#include "../../Algorithms/TripBased/Query/Types.h"
#include "../../Algorithms/UnionFind.h"
#include "../../Helpers/Types.h"
#include "../Graph/Graph.h"
#include "../RAPTOR/Data.h"
#include "Entities.h"
#include "UpdateTypes.h"
#include "../../Helpers/IO/Serialization.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <optional>
#include <cstdint>

namespace DynamicTimeTable {

// ---------------------------------------------------------
// The Dynamic TimeTable
// ---------------------------------------------------------
class Data {
public:
    // --- Initialization ---
    Data() = default;

    /**
     * @brief Initializes the dynamic timetable from static RAPTOR data.
     * Extracts all necessary data so RAPTOR::Data is no longer needed as a member.
     * Mints initial StopEventIds sequentially.
     */
    explicit inline Data(const RAPTOR::Data& raptorData) {
        transferGraph_ = raptorData.transferGraph;
        numberOfStops_ = raptorData.numberOfStops();

        RAPTOR::Data& rd = const_cast<RAPTOR::Data&>(raptorData);

        size_t numRoutes = rd.numberOfRoutes();
        routes_.resize(numRoutes);

        for (size_t r = 0; r < numRoutes; r++) {
            RouteId routeId(r);
            DynamicRoute& dynRoute = routes_[r];
            dynRoute.routeId = routeId;

            auto stopsOfRoute = rd.stopsOfRoute(routeId);
            for (auto stop : stopsOfRoute) {
                dynRoute.stopSequence.push_back(stop);
            }

            size_t numTrips = rd.numberOfTripsInRoute(routeId);
            dynRoute.trips.resize(numTrips);

            auto stopEvents = rd.stopEventsOfRoute(routeId);
            size_t numStops = stopsOfRoute.size();

            for (size_t tripIdx = 0; tripIdx < numTrips; tripIdx++) {
                DynamicTrip& dynTrip = dynRoute.trips[tripIdx];
                dynTrip.isActive = true;
                dynTrip.stopEvents.resize(numStops);

                StopEventId firstEventId(nextStopEventId_);
                firstEventToTripLocation_[firstEventId] = {routeId, tripIdx};

                for (size_t stopIdx = 0; stopIdx < numStops; stopIdx++) {
                    const RAPTOR::StopEvent& raptorEvent = stopEvents[tripIdx * numStops + stopIdx];
                    DynamicStopEvent& dynEvent = dynTrip.stopEvents[stopIdx];
                    dynEvent.id = StopEventId(nextStopEventId_);
                    dynEvent.stop = stopsOfRoute[stopIdx];
                    dynEvent.arrivalTime = raptorEvent.arrivalTime;
                    dynEvent.departureTime = raptorEvent.departureTime;
                    dynEvent.isSkipped = false;

                    eventToRoute_.push_back(routeId);
                    eventToFirstEvent_.push_back(firstEventId);
                    eventToStopIndex_.push_back(stopIdx);

                    nextStopEventId_++;
                }
            }
        }
    }

    // --- Dynamic Updates ---
    /**
     * @brief Applies a validated batch of GTFS-RT updates.
     * Enforces invariants (FIFO, stop sequences) and handles trip migrations if needed.
     */
    UpdateStatistics applyUpdates(const PendingUpdates& updates);

    /**
     * @brief Returns the summary of the latest batch of updates.
     * Consumed by the Transfer Update Stage.
     */
    const ChangeSummary& getLatestChanges() const;

    /**
     * @brief Clears the change summary. Called after transfers are updated.
     */
    void clearChangeSummary();

    // --- Export ---
    /**
     * @brief Flattens the dynamic structures into a highly-optimized, query-ready format.
     * Rebuilds mapping arrays, filters cancelled trips, and matches the memory layout
     * expected by TripBased::QueryData.
     * TODO: Can Cancelled/Skipped StopEventIds remain as holes to preserve
     * transfer graph stability or is a mapping needed (query and cell logic!!!)
     */
    TripBased::QueryData exportQueryData() const;

    // --- Basic Getters ---
    inline std::size_t numberOfStops() const { return numberOfStops_; }
    inline std::size_t numberOfRoutes() const { return routes_.size(); }
    inline std::size_t numberOfActiveTrips() const {
        std::size_t count = 0;
        for (const auto& route : routes_) {
            for (const auto& trip : route.trips) {
                if (trip.isActive) count++;
            }
        }
        return count;
    }
    inline std::size_t numberOfMintedStopEvents() const { return static_cast<std::size_t>(nextStopEventId_); }

    inline const DynamicRoute& getRoute(RouteId routeId) const { return routes_[routeId]; }

    // --- O(1) Resolution Helpers for Transfer Update ---
    inline RouteId getRouteOfEvent(StopEventId eventId) const { return eventToRoute_[eventId]; }
    inline StopEventId getFirstEventOfEvent(StopEventId eventId) const { return eventToFirstEvent_[eventId]; }
    inline size_t getStopIndexOfEvent(StopEventId eventId) const { return eventToStopIndex_[eventId]; }
    inline const DynamicTrip* getTripByFirstEvent(StopEventId firstEventId) const {
        auto it = firstEventToTripLocation_.find(firstEventId);
        if (it != firstEventToTripLocation_.end()) {
            return &routes_[it->second.first].trips[it->second.second];
        }
        return nullptr;
    }

    // --- Postprocessing & Partitioning ---
    void createCompactLayoutGraph();
    void applyGlobalIDs(const std::vector<uint64_t>& globalIds) noexcept;
    void readPartitionFile(const std::string& fileName);

    inline void serialize(const std::string& fileName) const noexcept {
        IO::serialize(fileName, routes_, nextStopEventId_, firstEventToTripLocation_, eventToRoute_, eventToFirstEvent_, eventToStopIndex_, transferGraph_, numberOfStops_, cellIds_, unionFind_, layoutGraph_);
    }

    inline void deserialize(const std::string& fileName) noexcept {
        IO::deserialize(fileName, routes_, nextStopEventId_, firstEventToTripLocation_, eventToRoute_, eventToFirstEvent_, eventToStopIndex_, transferGraph_, numberOfStops_, cellIds_, unionFind_, layoutGraph_);
    }

    void printInfo() const {
        std::cout << "DynamicTimeTable info\n";
        std::cout << "-----------------------------\n";
        std::cout << "Routes             : " << numberOfRoutes() << "\n";
        std::cout << "Active Trips       : " << numberOfActiveTrips() << "\n";
        std::cout << "Minted Stop Events : " << numberOfMintedStopEvents() << "\n";
    }

private:
    // --- Internal State ---

    // The core hierarchical data structure. Routes own their trips.
    std::vector<DynamicRoute> routes_;

    // Stable ID generator for StopEvents. Never decreases.
    StopEventId nextStopEventId_ = StopEventId(0);

    // Fast lookup to find the RouteId and Trip index from a first StopEventId.
    // Useful for applying updates to the correct DynamicTrip.
    std::unordered_map<StopEventId, std::pair<RouteId, size_t>> firstEventToTripLocation_;

    // Stable flat arrays for O(1) event resolution (even for cancelled events)
    std::vector<RouteId> eventToRoute_;
    std::vector<StopEventId> eventToFirstEvent_;
    std::vector<size_t> eventToStopIndex_;

    // Static topological data extracted from RAPTOR (never mutates in real-time)
    TransferGraph transferGraph_;
    std::size_t numberOfStops_ = 0;

    // Partitioning and Layout Graph
    std::vector<uint16_t> cellIds_;
    UnionFind unionFind_;
    StaticGraphWithWeightsAndCoordinates layoutGraph_;

    // Tracking for downstream subsystems
    ChangeSummary latestChanges_;

    // --- Update Helper Methods ---
    UpdateStatistics processCancellations(const std::vector<StopEventId>& cancellations);
    UpdateStatistics processModifications(const std::vector<std::pair<StopEventId, std::vector<StopModification>>>& modifications);
    UpdateStatistics processAdditions(const std::vector<AddedTripInfo>& additions);

    /**
     * @brief Checks if a modified trip violates the FIFO property of its current route.
     */
    bool checkFifoViolation(RouteId routeId, size_t tripIndex) const;

    /**
     * @brief Resolves FIFO violations by moving the trip to a compatible route
     * or creating a new route variant. Identifies violating trips by their first StopEventId.
     */
    void resolveFifoViolations(const std::vector<StopEventId>& violatingTrips);

    /**
     * @brief Assigns an extracted or new trip to the optimal route, creating a new route
     * if no compatible sequence/FIFO-slot is found.
     * Takes ownership of the DynamicTrip object.
     */
    bool assignTripToOptimalRoute(DynamicTrip&& trip, const std::vector<StopId>& stopSequence);

    /**
     * @brief Generates a new internal RouteId and registers a new DynamicRoute for sequence variants.
     */
    RouteId createNewRouteVariant(const std::vector<StopId>& stopSequence);

    /**
     * @brief Refreshes the firstEventToTripLocation_ map for a specific route.
     * Often needed after trips are sorted, added, or removed.
     */
    void rebuildTripLocationIndexForRoute(RouteId routeId);
};

} // namespace DynamicTimeTable
