#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../Algorithms/UnionFind.h"
#include "../../Helpers/IO/Serialization.h"
#include "../../Helpers/Types.h"
#include "../../Helpers/Vector/Vector.h"
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

    // --- Memory footprint (structure byteSize) ---
    // Long-term persistent workspace: routes (each with inner stopSequence/trips
    // vectors), trips, events, the event->trip map, min transfer times, cell ids
    // and the static transfer graph. The sequence-hash index is approximated.
    // `capacity` selects size() (logical) vs capacity() (reserved) accounting.
    long long dataBytes(bool capacity) const noexcept {
        auto v = [&](const auto& vec) {
            return capacity ? Vector::memoryUsageInBytes(vec) : Vector::byteSize(vec);
        };
        long long r = static_cast<long long>(sizeof(PersistentRoute)) *
                      (capacity ? static_cast<long long>(routes_.capacity()) : static_cast<long long>(routes_.size()));
        for (const auto& route : routes_) r += v(route.stopSequence) + v(route.trips);
        r += v(trips_) + v(events_) + v(eventToTrip_) + v(minTransferTimes_) + v(cellIds_);
        r += capacity ? transferGraph_.memoryUsageInBytes() : transferGraph_.byteSize();
        // Approximate the sequence-hash multimap (bucket entries + payload vectors).
        for (const auto& [key, bucket] : routesBySequenceHash_) {
            r += static_cast<long long>(sizeof(std::size_t) + sizeof(void*) * 2) + v(bucket);
        }
        return r;
    }

    long long byteSize() const noexcept { return dataBytes(false); }
    long long memoryUsageInBytes() const noexcept { return dataBytes(true); }

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

    // --- Partitioning / cell hierarchy ---
    //
    // The partition is what makes TREX possible: `cellIds_[stop]` is a 16-bit hierarchical
    // cell id whose bit prefixes encode the nesting, so "same cell at level L" is
    // `!((a ^ b) >> L)`. RT updates never add stops or repartition, so all of this is
    // computed once and then held invariant across every timetable update.
    //
    // These mirror TREXData::createCompactLayoutGraph / readPartitionFile / applyGlobalIDs
    // (DataStructures/TREX/TREXData.h) so a partition file produced for the static TREX
    // instance of the same network can be applied here unchanged. Only the union-find
    // contraction is ported: the layout graph exists solely to *emit* a METIS/KaHyPar
    // instance, which stays a static-side responsibility.

    uint16_t getCellIdOfStop(const StopId stop) const noexcept {
        AssertMsg(static_cast<std::size_t>(stop) < cellIds_.size(), "Stop is out of bounds!");
        return cellIds_[stop];
    }

    const std::vector<uint16_t>& cellIds() const noexcept { return cellIds_; }

    bool hasPartition() const noexcept { return numberOfLevels_ > 0 && !cellIds_.empty(); }

    int getNumberOfLevels() const noexcept { return numberOfLevels_; }

    void setNumberOfLevels(const int levels) noexcept { numberOfLevels_ = levels; }

    /**
     * Recover the level count from the cell ids themselves. Cell ids are hierarchical bit
     * prefixes with two cells per level, so the number of levels is the bit width of the
     * largest id. Keeping this derivable means the partition needs no extra serialized
     * field and existing dynamic.binary files stay loadable.
     */
    void deriveNumberOfLevels() noexcept {
        uint16_t maxCellId = 0;
        for (const uint16_t cellId : cellIds_) maxCellId = std::max(maxCellId, cellId);
        numberOfLevels_ = static_cast<int>(std::bit_width(maxCellId));
    }

    /**
     * Contract every footpath-connected component into one union-find representative, so a
     * cut can never separate two stops joined by a footpath. `componentWeight_[rep]` is the
     * component size (0 for non-representatives), used to sanity-check applyGlobalIDs.
     */
    void createCompactLayoutGraph() {
        unionFind_.reset(static_cast<int>(numberOfStops_));

        for (const auto [edge, from] : transferGraph_.edgesWithFromVertex()) {
            const Vertex toStop = transferGraph_.get(ToVertex, edge);
            unionFind_(from, toStop);
        }

        componentWeight_.assign(numberOfStops_, 0);
        for (std::size_t i = 0; i < numberOfStops_; ++i) {
            ++componentWeight_[static_cast<std::size_t>(unionFind_(static_cast<int>(i)))];
        }
    }

    /**
     * @param globalIds cell id per union-find representative, as emitted by the partitioner.
     */
    void applyGlobalIDs(const std::vector<uint64_t>& globalIds) {
        if (componentWeight_.size() != numberOfStops_) createCompactLayoutGraph();
        cellIds_.assign(numberOfStops_, 0);

        for (std::size_t i = 0; i < numberOfStops_; ++i) {
            const int representative = unionFind_(static_cast<int>(i));
            AssertMsg(static_cast<std::size_t>(representative) < globalIds.size(), "unionFind is out of bounds!");
            AssertMsg(componentWeight_[static_cast<std::size_t>(representative)] > 0,
                      "The corresponding component weight is zero?");
            cellIds_[i] = static_cast<uint16_t>(globalIds[static_cast<std::size_t>(representative)]);
        }

        AssertMsg(assertNoCutTransfers(), "Footpath has been cut!");
        deriveNumberOfLevels();
    }

    void readPartitionFile(const std::string& fileName) {
        std::vector<uint64_t> globalIds(numberOfStops_, 0);
        std::fstream file(fileName);

        if (!file.is_open()) {
            std::cerr << "Unable to open the file: " << fileName << std::endl;
            return;
        }

        uint64_t globalId(0);
        std::size_t index(0);
        while (file >> globalId) {
            if (index >= globalIds.size()) break;
            globalIds[index] = globalId;
            ++index;
        }
        file.close();
        std::cout << "Read " << index << " many IDs!" << std::endl;

        applyGlobalIDs(globalIds);
    }

    /**
     * Every footpath must stay inside one cell -- otherwise a walk could leave the cell
     * without crossing a border stop event, and the customization would miss it.
     */
    bool assertNoCutTransfers() const noexcept {
        for (const auto [edge, from] : transferGraph_.edgesWithFromVertex()) {
            const Vertex toStop = transferGraph_.get(ToVertex, edge);
            if (cellIds_[from] != cellIds_[toStop]) return false;
        }
        return true;
    }

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
        deriveNumberOfLevels();
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

    // Partitioning / cell hierarchy. Invariant under RT updates.
    std::vector<uint16_t> cellIds_;
    int numberOfLevels_ = 0;
    UnionFind unionFind_;
    // Size of each footpath-connected component, indexed by its union-find representative
    // (0 for non-representatives). Derived state, rebuilt by createCompactLayoutGraph().
    std::vector<std::uint32_t> componentWeight_;
    StaticGraphWithWeightsAndCoordinates layoutGraph_;

    ChangeSummary latestChanges_;
};

}  // namespace DynamicTimeTable
