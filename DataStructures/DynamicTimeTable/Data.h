#pragma once

#include <cstddef>
#include <optional>
#include <unordered_map>
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

/**
 * DynamicTimeTable
 *
 * Persistent ids are dense and directly index the corresponding vectors.
 * Trips own a contiguous stop-event block.
 * Routes have an immutable stop sequence; if a trip's effective sequence changes, it migrates.
 */
class Data {
public:
    Data() = default;

    explicit Data(const RAPTOR::Data& raptorData) { importFromRaptor(raptorData); }

    void importFromRaptor(const RAPTOR::Data& raptorData) {
        transferGraph_ = raptorData.transferGraph;
        numberOfStops_ = raptorData.numberOfStops();

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
        rd.useImplicitArrivalBufferTimes();
        rd.useImplicitDepartureBufferTimes();

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

    // --- Updates ---

    struct UpdateContext {
        ChangeSummary summary;
        std::vector<PersistentTripId> extractionQueue;
        // Group modified trips by route to optimize pairwise FIFO checks
        std::unordered_map<PersistentRouteId, std::vector<PersistentTripId>> modifiedTripsByRoute;
    };

    /**
     * Applies a batch of updates (cancellations, modifications, additions)
     * to the dynamic timetable, ensuring structural and FIFO invariants are maintained.
     *
     * Input: A sanitized PendingUpdates object containing the batch.
     * Output: UpdateStatistics detailing the success/failure of the batch.
     * Side Effects: Modifies routes, trips, and events in place. Emits changes to latestChanges_.
     */
    UpdateStatistics applyUpdates(const PendingUpdates& updates) {
        latestChanges_.clear();
        UpdateStatistics stats{};

        if (!updates.hasUpdates()) return stats;

        UpdateContext context;

        // stats += processCancellations(updates, context);
        // stats += processModifications(updates, context);
        // enforceFifo(context);
        // stats += processInsertions(updates, context);

        // latestChanges_ = std::move(context.summary);

        return stats;
    }

    const ChangeSummary& getLatestChanges() const { return latestChanges_; }

    void clearChangeSummary() { latestChanges_.clear(); }

    // --- Export ---
    // #TODO

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
                      numberOfStops_, cellIds_, unionFind_, layoutGraph_, latestChanges_);
    }

    void deserialize(const std::string& fileName) noexcept {
        IO::deserialize(fileName, routes_, trips_, events_, routesBySequenceHash_, eventToTrip_, transferGraph_,
                        numberOfStops_, cellIds_, unionFind_, layoutGraph_, latestChanges_);
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

    /**
     * Phase 1: Cancellations
     * Input: pendingUpdates.cancellations
     * Action: Mark trip isActive = false, remove from route.trips.
     * Output: Add tripId to context.summary.cancelledTrips.
     */
    // UpdateStatistics processCancellations(const PendingUpdates& updates, UpdateContext& context) {
    //     return UpdateStatistics{};
    // }

    /**
     * Phase 2: Modifications
     * Input: pendingUpdates.modifications
     * Action: Apply in-place time/status changes and identify structural breaks.
     * Logic Routing:
     *   - If isSkipped changed -> extractTrip(...)
     *   - Else -> Group modified trip by its route in context.modifiedTripsByRoute.
     * Output: Populates context.summary.modifiedTrips and delayedArrivals.
     */
    // UpdateStatistics processModifications(const PendingUpdates& updates, UpdateContext& context) {
    //     return UpdateStatistics{};
    // }

    /**
     * Phase 3: Enforce FIFO
     * Input: context.modifiedTripsByRoute
     * Action: For each affected route, run deterministic greedy algorithm to resolve FIFO violations.
     *         Because trips part of a FIFO violation must include at least one modified trip,
     *         the algorithm focuses around the modified trips. Trips with the "most violations"
     *         are prioritized for extraction.
     * Logic Routing: If a trip violates FIFO, remove it and call extractTrip(...)
     * Output: Appends to context.extractionQueue.
     */
    // void enforceFifo(UpdateContext& context) {}

    /**
     * Phase 4: Insertions
     * Input: pendingUpdates.additions AND context.extractionQueue
     * Action: Find or create a FIFO-compatible route for each trip and insert.
     * Output: Add to context.summary.addedTrips.
     */
    // UpdateStatistics processInsertions(const PendingUpdates& updates, UpdateContext& context) {
    //     return UpdateStatistics{};
    // }

    /**
     * Helper: Extract Trip
     * Action: Removes trip from its current route.trips, records old route ID, pushes to extractionQueue.
     */
    // void extractTrip(const PersistentTripId tripId, UpdateContext& context) {}

    /**
     * Helper: Find Compatible Route
     * Action: Hashes stopSequence, checks stopSequenceEquals and isFifoCompatible. Returns matched or new RouteId.
     */
    // PersistentRouteId findCompatibleRoute(const std::vector<StopId>& stopSequence, const PersistentTripId tripId) {
    //     return PersistentRouteId(0);
    // }

    /**
     * Helper: Is FIFO Compatible
     * Action: Checks if inserting times of tripId into routeId creates a FIFO inversion.
     */
    // bool isFifoCompatible(const PersistentRouteId routeId, const PersistentTripId tripId) const {
    //     return true;
    // }

private:
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

    // Partitioning/Layout (kept for compatibility with existing code paths)
    std::vector<uint16_t> cellIds_;
    UnionFind unionFind_;
    StaticGraphWithWeightsAndCoordinates layoutGraph_;

    ChangeSummary latestChanges_;
};

}  // namespace DynamicTimeTable
