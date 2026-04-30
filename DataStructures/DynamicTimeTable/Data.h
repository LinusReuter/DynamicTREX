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

namespace DynamicTB {
    struct DynamicQueryData;
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
    friend struct ::DynamicTB::DynamicQueryData;

public:
    Data() = default;

    explicit Data(const std::string& fileName) { deserialize(fileName); }

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

        stats += processCancellations(updates, context);
        stats += processModifications(updates, context);
        enforceFifo(context);
        stats += processInsertions(updates, context);

        latestChanges_ = std::move(context.summary);

        return stats;
    }

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
    UpdateStatistics processCancellations(const PendingUpdates& updates, UpdateContext& context) {
        UpdateStatistics stats{};
        stats.cancellations = updates.cancellations.size();
        stats.totalUpdates += updates.cancellations.size();

        for (const PersistentTripId tripId : updates.cancellations) {
            if (!isTrip(tripId)) {
                stats.failedUpdates++;
                continue;
            }

            PersistentTrip& trip = trips_[tripId];
            if (!trip.isActive) {
                stats.failedUpdates++;
                continue;
            }

            const PersistentRouteId oldRoute = trip.route;
            trip.isActive = false;

            if (isRoute(oldRoute)) {
                auto& list = routes_[oldRoute].trips;
                // Preserve chronological order of remaining trips
                list.erase(std::remove(list.begin(), list.end(), tripId), list.end());
            }

            context.summary.cancelledTrips.push_back({tripId, oldRoute});
            stats.successfulUpdates++;
        }

        return stats;
    }

    /**
     * Phase 2: Modifications
     * Input: pendingUpdates.modifications
     * Action: Apply in-place time/status changes and identify structural breaks.
     * Logic Routing:
     *   - If isSkipped changed -> extractTrip(...)
     *   - Else -> Group modified trip by its route in context.modifiedTripsByRoute.
     * Output: Populates context.summary.modifiedTrips and delayedArrivals.
     */
    UpdateStatistics processModifications(const PendingUpdates& updates, UpdateContext& context) {
        UpdateStatistics stats{};
        stats.modifications = updates.modifications.size();
        stats.totalUpdates += updates.modifications.size();

        for (const auto& [tripId, mods] : updates.modifications) {
            if (!isTrip(tripId)) {
                stats.failedUpdates++;
                continue;
            }

            PersistentTrip& trip = trips_[tripId];
            if (!trip.isActive) {
                stats.failedUpdates++;
                continue;
            }

            bool structural = false;
            bool delayedArrivals = false;

            for (const StopModification& m : mods) {
                const std::size_t idx = static_cast<std::size_t>(m.stopIndex);
                if (idx >= trip.numberOfEvents) {
                    structural = true;
                    continue;
                }

                const std::size_t first = static_cast<std::size_t>(trip.firstEvent);
                const PersistentStopEventId eventId(first + idx);
                if (!isEvent(eventId)) {
                    structural = true;
                    continue;
                }

                PersistentStopEvent& e = events_[eventId];
                const Time oldArr = e.arrivalTime;

                if (m.newArrivalTime != noTime) e.arrivalTime = m.newArrivalTime;
                if (m.newDepartureTime != noTime) e.departureTime = m.newDepartureTime;

                // We expect the upstream stage to have sanitized the data, but we assert just in case.
                AssertMsg(e.arrivalTime <= e.departureTime, "Logically invalid modification: arrival > departure");

                if (m.isSkipped) {
                    e.isSkipped = true;
                    structural = true;
                }

                if (e.arrivalTime > oldArr) delayedArrivals = true;

                context.summary.modifiedEvents.push_back(eventId);
            }

            if (delayedArrivals) {
                context.summary.tripsWithDelayedArrivals.push_back(tripId);
            }

            if (structural) {
                extractTrip(tripId, context);
            } else {
                context.modifiedTripsByRoute[trip.route].push_back(tripId);
                stats.successfulUpdates++;
            }
        }

        // Restore chronological sort order for affected routes.
        // NOTE: std::stable_sort is robust and easy. However, if no extractions occur and only
        // small time changes happen, local swaps (e.g. insertion sort) would likely be more efficient.
        for (const auto& [routeId, modTrips] : context.modifiedTripsByRoute) {
            auto& list = routes_[routeId].trips;
            std::stable_sort(list.begin(), list.end(), [this](PersistentTripId a, PersistentTripId b) {
                return getFirstDepartureTime(a) < getFirstDepartureTime(b);
            });
        }

        return stats;
    }

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
    void enforceFifo(UpdateContext& context) {
        for (auto& [routeId, modTrips] : context.modifiedTripsByRoute) {
            auto& list = routes_[routeId].trips;
            if (list.size() < 2) continue;

            // Map tripId -> set of violating tripIds
            std::unordered_map<PersistentTripId, std::unordered_set<PersistentTripId>> violations;

            for (PersistentTripId m_id : modTrips) {
                auto it = std::find(list.begin(), list.end(), m_id);
                if (it == list.end()) continue;
                std::ptrdiff_t idx = std::distance(list.begin(), it);

                // Check backwards
                for (std::ptrdiff_t i = idx - 1; i >= 0; --i) {
                    if (checkFifoViolation(list[i], m_id)) {
                        violations[m_id].insert(list[i]);
                        violations[list[i]].insert(m_id);
                    } else {
                        // The list is sorted chronologically. If list[i] is completely and safely BEFORE m_id,
                        // we can stop scanning backwards, as transitive property holds.
                        break;
                    }
                }
                // Check forwards
                for (std::ptrdiff_t i = idx + 1; i < static_cast<std::ptrdiff_t>(list.size()); ++i) {
                    if (checkFifoViolation(m_id, list[i])) {
                        violations[m_id].insert(list[i]);
                        violations[list[i]].insert(m_id);
                    } else {
                        break;
                    }
                }
            }

            // Greedily extract worst offenders
            while (!violations.empty()) {
                PersistentTripId worstTrip = noPersistentTripId;
                std::size_t maxDegree = 0;

                for (const auto& [t, vSet] : violations) {
                    if (vSet.size() > maxDegree) {
                        maxDegree = vSet.size();
                        worstTrip = t;
                    } else if (vSet.size() == maxDegree && maxDegree > 0) {
                        // Deterministic tie-break
                        if (static_cast<std::size_t>(t) > static_cast<std::size_t>(worstTrip)) {
                            worstTrip = t;
                        }
                    }
                }

                if (maxDegree == 0) break;

                extractTrip(worstTrip, context);

                // The trip was removed from routes_[routeId].trips inside extractTrip.
                // Now clean up the localized violation graph.
                const auto& neighbors = violations[worstTrip];
                for (PersistentTripId n : neighbors) {
                    violations[n].erase(worstTrip);
                    if (violations[n].empty()) {
                        violations.erase(n);
                    }
                }
                violations.erase(worstTrip);
            }
        }
    }

    /**
     * Phase 4: Insertions
     * Input: pendingUpdates.additions AND context.extractionQueue
     * Action: Find or create a FIFO-compatible route for each trip and insert.
     * Output: Add to context.summary.addedTrips.
     */
    UpdateStatistics processInsertions(const PendingUpdates& updates, UpdateContext& context) {
        UpdateStatistics stats{};
        stats.additions = updates.additions.size();
        stats.totalUpdates += updates.additions.size();

        // 1. Process brand new additions
        for (const AddedTripInfo& add : updates.additions) {
            if (add.stopSequence.empty() || add.arrivalTimes.size() != add.stopSequence.size() ||
                add.departureTimes.size() != add.stopSequence.size()) {
                stats.failedUpdates++;
                continue;
            }

            const PersistentTripId newTripId(trips_.size());
            PersistentTrip trip;
            trip.isActive = true;
            trip.firstEvent = PersistentStopEventId(events_.size());
            trip.numberOfEvents = static_cast<std::uint32_t>(add.stopSequence.size());

            for (std::size_t i = 0; i < add.stopSequence.size(); i++) {
                PersistentStopEvent e;
                e.stop = add.stopSequence[i];
                e.arrivalTime = add.arrivalTimes[i];
                e.departureTime = add.departureTimes[i];
                e.isSkipped = false;
                events_.push_back(e);
                eventToTrip_.push_back(newTripId);
            }

            PersistentRouteId routeId = findCompatibleRoute(add.stopSequence, newTripId);
            trip.route = routeId;
            trips_.push_back(trip);

            insertTripChronologically(routeId, newTripId);
            context.summary.addedTrips.push_back(newTripId);
            stats.successfulUpdates++;
        }

        // 2. Process re-insertions (trips extracted due to structural/FIFO issues)
        for (const PersistentTripId tripId : context.extractionQueue) {
            if (!isTrip(tripId)) continue;
            PersistentTrip& trip = trips_[tripId];
            if (!trip.isActive) continue;

            std::vector<StopId> effectiveSeq = getEffectiveSequence(tripId);
            if (effectiveSeq.empty()) {
                trip.isActive = false;
                continue;
            }

            PersistentRouteId routeId = findCompatibleRoute(effectiveSeq, tripId);
            trip.route = routeId;
            insertTripChronologically(routeId, tripId);
            context.summary.addedTrips.push_back(tripId);
        }

        return stats;
    }

    /**
     * Helper: Extract Trip
     * Action: Removes trip from its current route.trips, records old route ID, pushes to extractionQueue.
     */
    void extractTrip(const PersistentTripId tripId, UpdateContext& context) {
        PersistentTrip& trip = trips_[tripId];
        const PersistentRouteId oldRoute = trip.route;

        if (isRoute(oldRoute)) {
            auto& list = routes_[oldRoute].trips;
            list.erase(std::remove(list.begin(), list.end(), tripId), list.end());
        }

        context.summary.cancelledTrips.push_back({tripId, oldRoute});
        context.extractionQueue.push_back(tripId);
    }

    /**
     * Helper: Find Compatible Route
     * Action: Hashes stopSequence, checks stopSequenceEquals and isFifoCompatible. Returns matched or new RouteId.
     */
    PersistentRouteId findCompatibleRoute(const std::vector<StopId>& stopSequence, const PersistentTripId tripId) {
        const std::size_t h = hashStopSequence(stopSequence);
        auto it = routesBySequenceHash_.find(h);

        if (it != routesBySequenceHash_.end()) {
            for (const PersistentRouteId candidate : it->second) {
                if (!isRoute(candidate)) continue;
                if (stopSequenceEquals(routes_[candidate].stopSequence, stopSequence)) {
                    if (isFifoCompatible(candidate, tripId)) {
                        return candidate;
                    }
                }
            }
        }

        const PersistentRouteId newId(routes_.size());
        PersistentRoute r;
        r.routeId = newId;
        r.stopSequence = stopSequence;

        routes_.push_back(std::move(r));
        routesBySequenceHash_[h].push_back(newId);

        return newId;
    }

    /**
     * Helper: Is FIFO Compatible
     * Action: Checks if inserting times of tripId into routeId creates a FIFO inversion.
     */
    bool isFifoCompatible(const PersistentRouteId routeId, const PersistentTripId tripId) const {
        const auto& list = routes_[routeId].trips;
        if (list.empty()) return true;

        Time dep = getFirstDepartureTime(tripId);
        auto it = std::upper_bound(list.begin(), list.end(), dep,
                                   [this](Time val, PersistentTripId t) { return val < getFirstDepartureTime(t); });

        // Check predecessor
        if (it != list.begin()) {
            PersistentTripId pred = *(it - 1);
            if (checkFifoViolation(pred, tripId)) return false;
        }
        // Check successor
        if (it != list.end()) {
            PersistentTripId succ = *it;
            if (checkFifoViolation(tripId, succ)) return false;
        }

        return true;
    }

    // --- Internal Update Pipeline Helpers ---

    Time getFirstDepartureTime(const PersistentTripId tripId) const {
        const PersistentTrip& trip = trips_[tripId];
        for (std::uint32_t i = 0; i < trip.numberOfEvents; i++) {
            const PersistentStopEvent& e = events_[trip.firstEvent + i];
            if (!e.isSkipped) return e.departureTime;
        }
        // Should not be reachable for structurally sound trips, but provide a fallback
        return Time(std::numeric_limits<int>::max());
    }

    std::vector<StopId> getEffectiveSequence(const PersistentTripId tripId) const {
        std::vector<StopId> seq;
        const PersistentTrip& trip = trips_[tripId];
        seq.reserve(trip.numberOfEvents);
        for (std::uint32_t i = 0; i < trip.numberOfEvents; i++) {
            const PersistentStopEvent& e = events_[trip.firstEvent + i];
            if (!e.isSkipped) seq.push_back(e.stop);
        }
        return seq;
    }

    void insertTripChronologically(const PersistentRouteId routeId, const PersistentTripId tripId) {
        auto& list = routes_[routeId].trips;
        Time dep = getFirstDepartureTime(tripId);
        auto it = std::upper_bound(list.begin(), list.end(), dep,
                                   [this](Time val, PersistentTripId t) { return val < getFirstDepartureTime(t); });
        list.insert(it, tripId);
    }

    bool checkFifoViolation(const PersistentTripId a, const PersistentTripId b) const {
        // Assume 'a' is scheduled to run chronologically before 'b'.
        const PersistentTrip& ta = trips_[a];
        const PersistentTrip& tb = trips_[b];

        std::uint32_t ia = 0, ib = 0;
        while (ia < ta.numberOfEvents && ib < tb.numberOfEvents) {
            const PersistentStopEvent& ea = events_[ta.firstEvent + ia];
            if (ea.isSkipped) {
                ia++;
                continue;
            }
            const PersistentStopEvent& eb = events_[tb.firstEvent + ib];
            if (eb.isSkipped) {
                ib++;
                continue;
            }

            // TODO: Equal times are currently considered valid (not a violation).
            // Verify if this holds true for all downstream consumers.
            if (ea.arrivalTime > eb.arrivalTime || ea.departureTime > eb.departureTime) {
                return true;  // Violation!
            }
            ia++;
            ib++;
        }
        return false;
    }

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
