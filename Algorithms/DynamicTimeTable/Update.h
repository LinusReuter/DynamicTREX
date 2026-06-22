#pragma once

#include <algorithm>
#include <limits>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../DataStructures/DynamicTimeTable/Data.h"
#include "../../DataStructures/DynamicTimeTable/UpdateTypes.h"

namespace DynamicTimeTable::Algo {

struct RouteTripPair {
    PersistentRouteId routeId;
    PersistentTripId tripId;
};

struct UpdateContext {
    ChangeSummary summary;
    std::vector<PersistentTripId> extractionQueue;
    // Group modified trips by route to optimize pairwise FIFO checks
    std::unordered_map<PersistentRouteId, std::vector<PersistentTripId>> modifiedTripsByRoute;

    std::vector<RouteTripPair> tripsAfterExtractedTrips;
};

struct UpdatePipeline {
    static UpdateStatistics applyUpdates(Data& data, const PendingUpdates& updates) {
        data.latestChanges_.clear();
        UpdateStatistics stats{};

        if (!updates.hasUpdates()) return stats;

        UpdateContext context;

        stats += processCancellations(data, updates, context);
        stats += processModifications(data, updates, context);
        enforceFifo(data, context);
        stats += processInsertions(data, updates, context);

        finalizeChangeSummery(data, context);
        data.latestChanges_ = std::move(context.summary);

        return stats;
    }

private:
    static std::span<const PersistentStopEvent> getTripEvents(const Data& data, const PersistentTripId tripId) {
        const PersistentTrip& trip = data.trips_[tripId];
        return {&data.events_[trip.firstEvent], trip.numberOfEvents};
    }

    static UpdateStatistics processCancellations(Data& data, const PendingUpdates& updates, UpdateContext& context) {
        UpdateStatistics stats{};
        stats.cancellations = updates.cancellations.size();
        stats.totalUpdates += updates.cancellations.size();

        for (const PersistentTripId tripId : updates.cancellations) {
            if (!data.isTrip(tripId)) {
                stats.failedUpdates++;
                continue;
            }

            PersistentTrip& trip = data.trips_[tripId];
            if (!trip.isActive) {
                stats.failedUpdates++;
                continue;
            }

            const PersistentRouteId oldRoute = trip.route;
            std::vector<PersistentStopEventId> activeEvents = collectActiveEvents(data, tripId);

            trip.isActive = false;
            trip.route = noPersistentRouteId;

            if (data.isRoute(oldRoute)) {
                auto& list = data.routes_[oldRoute].trips;
                if (auto it = std::ranges::find(list, tripId); it != list.end() && std::next(it) != list.end()) {
                    context.tripsAfterExtractedTrips.push_back({oldRoute, *std::next(it)});
                }
                std::erase(list, tripId);
            }

            context.summary.cancelledTrips.push_back(makeCancelledTripInfo(tripId, std::move(activeEvents)));
            stats.successfulUpdates++;
        }

        return stats;
    }

    static UpdateStatistics processModifications(Data& data, const PendingUpdates& updates, UpdateContext& context) {
        UpdateStatistics stats{};
        stats.modifications = updates.modifications.size();
        stats.totalUpdates += updates.modifications.size();

        for (const auto& [tripId, mods] : updates.modifications) {
            if (!data.isTrip(tripId)) {
                stats.failedUpdates++;
                continue;
            }

            PersistentTrip& trip = data.trips_[tripId];
            if (!trip.isActive) {
                stats.failedUpdates++;
                continue;
            }

            bool structural = false;
            bool delayedArrivals = false;
            std::vector<PersistentStopEventId> preActiveEvents;

            // Pre-scan to detect structural changes before mutating skip flags.
            for (const StopModification& m : mods) {
                const auto idx = static_cast<std::size_t>(m.stopIndex);
                if (idx >= trip.numberOfEvents) {
                    structural = true;
                    continue;
                }

                const auto first = static_cast<std::size_t>(trip.firstEvent);
                const PersistentStopEventId eventId(first + idx);
                if (!data.isEvent(eventId)) {
                    structural = true;
                    continue;
                }

                const PersistentStopEvent& e = data.events_[eventId];
                if (m.isSkipped != e.isSkipped) {
                    structural = true;
                }
            }

            if (structural) {
                preActiveEvents = collectActiveEvents(data, tripId);
            }

            for (const StopModification& m : mods) {
                const auto idx = static_cast<std::size_t>(m.stopIndex);
                if (idx >= trip.numberOfEvents) {
                    structural = true;
                    continue;
                }

                const auto first = static_cast<std::size_t>(trip.firstEvent);
                const PersistentStopEventId eventId(first + idx);
                if (!data.isEvent(eventId)) {
                    structural = true;
                    continue;
                }

                PersistentStopEvent& e = data.events_[eventId];
                const Time oldArr = e.arrivalTime;

                if (m.newArrivalTime != noTime) e.arrivalTime = m.newArrivalTime;
                if (m.newDepartureTime != noTime) e.departureTime = m.newDepartureTime;

                // We expect the upstream stage to have sanitized the data, but we assert just in case.
                AssertMsg(e.arrivalTime <= e.departureTime, "Logically invalid modification: arrival > departure");

                if (m.isSkipped != e.isSkipped) {
                    e.isSkipped = m.isSkipped;
                    structural = true;
                }

                if (e.arrivalTime > oldArr) delayedArrivals = true;

                context.summary.modifiedEvents.push_back(eventId);
            }

            if (delayedArrivals) {
                context.summary.tripsWithDelayedArrivals.push_back(tripId);
            }

            if (structural) {
                extractTrip(data, tripId, context, std::move(preActiveEvents));
            } else {
                context.modifiedTripsByRoute[trip.route].push_back(tripId);
                stats.successfulUpdates++;
            }
        }

        return stats;
    }

    static void enforceFifo(Data& data, UpdateContext& context) {
        for (auto& [routeId, modTrips] : context.modifiedTripsByRoute) {
            auto& list = data.routes_[routeId].trips;
            if (list.size() < 2) continue;

            // Map tripId -> set of violating tripIds
            std::unordered_map<PersistentTripId, std::unordered_set<PersistentTripId>> violations;

            for (PersistentTripId m_id : modTrips) {
                auto it = std::ranges::find(list, m_id);
                if (it == list.end()) continue;
                std::ptrdiff_t idx = std::distance(list.begin(), it);

                // Check backwards (list[i] is expected to be strictly BEFORE m_id)
                for (std::ptrdiff_t i = idx - 1; i >= 0; --i) {
                    int cmp = compareFifo(data, list[i], m_id);

                    if (cmp != -1) {
                        // cmp == 0 (crossover) or cmp == 1 (inversion) are both structural violations
                        violations[m_id].insert(list[i]);
                        violations[list[i]].insert(m_id);
                    } else {
                        // list[i] is completely and safely BEFORE m_id.
                        // As the list maintains chronological order, we can safely stop scanning backwards.
                        break;
                    }
                }

                // Check forwards (m_id is expected to be strictly BEFORE list[i])
                for (std::ptrdiff_t i = idx + 1; i < static_cast<std::ptrdiff_t>(list.size()); ++i) {
                    int cmp = compareFifo(data, m_id, list[i]);

                    if (cmp != -1) {
                        // cmp == 0 (crossover) or cmp == 1 (inversion) are both structural violations
                        violations[m_id].insert(list[i]);
                        violations[list[i]].insert(m_id);
                    } else {
                        // m_id is completely and safely BEFORE list[i].
                        // We can safely stop scanning forwards.
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

                std::vector<PersistentStopEventId> preActiveEvents = collectActiveEvents(data, worstTrip);
                extractTrip(data, worstTrip, context, std::move(preActiveEvents));

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

    static UpdateStatistics processInsertions(Data& data, const PendingUpdates& updates, UpdateContext& context) {
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

            const PersistentTripId newTripId(data.trips_.size());
            PersistentTrip trip;
            trip.isActive = true;
            trip.firstEvent = PersistentStopEventId(data.events_.size());
            trip.numberOfEvents = static_cast<std::uint32_t>(add.stopSequence.size());

            for (std::size_t i = 0; i < add.stopSequence.size(); i++) {
                PersistentStopEvent e;
                e.stop = add.stopSequence[i];
                e.arrivalTime = add.arrivalTimes[i];
                e.departureTime = add.departureTimes[i];
                e.isSkipped = false;
                data.events_.push_back(e);
                data.eventToTrip_.push_back(newTripId);
            }

            auto [routeId, idx] = findCompatibleRoute(data, add.stopSequence, getTripEvents(data, newTripId));
            trip.route = routeId;
            data.trips_.push_back(trip);
            auto& list = data.routes_[routeId].trips;
            list.insert(list.begin() + idx, newTripId);

            context.summary.addedTrips.push_back(newTripId);
            stats.successfulUpdates++;
        }

        // 2. Process re-insertions (trips extracted due to structural/FIFO issues)
        for (const PersistentTripId tripId : context.extractionQueue) {
            if (!data.isTrip(tripId)) continue;
            PersistentTrip& trip = data.trips_[tripId];
            if (!trip.isActive) continue;

            std::vector<StopId> effectiveSeq = getEffectiveSequence(data, tripId);
            if (effectiveSeq.empty()) {
                trip.isActive = false;
                continue;
            }

            auto [routeId, idx] = findCompatibleRoute(data, effectiveSeq, getTripEvents(data, tripId));
            trip.route = routeId;
            auto& list = data.routes_[routeId].trips;
            list.insert(list.begin() + idx, tripId);
            context.summary.addedTrips.push_back(tripId);
        }

        return stats;
    }

    static void extractTrip(Data& data, const PersistentTripId tripId, UpdateContext& context,
                            std::vector<PersistentStopEventId>&& preActiveEvents) {
        PersistentTrip& trip = data.trips_[tripId];
        const PersistentRouteId oldRoute = trip.route;
        trip.route = noPersistentRouteId;

        if (data.isRoute(oldRoute)) {
            auto& list = data.routes_[oldRoute].trips;

            if (const auto it = std::ranges::find(list, tripId); it != list.end() && std::next(it) != list.end()) {
                context.tripsAfterExtractedTrips.push_back({oldRoute, *std::next(it)});
            }
            std::erase(list, tripId);
        }

        context.summary.cancelledTrips.push_back(makeCancelledTripInfo(tripId, std::move(preActiveEvents)));
        context.extractionQueue.push_back(tripId);
    }

    static CancelledTripInfo makeCancelledTripInfo(const PersistentTripId tripId,
                                                   std::vector<PersistentStopEventId>&& activeEvents) {
        CancelledTripInfo info;
        info.tripId = tripId;
        info.eventsOfCancelledTrips = std::move(activeEvents);
        return info;
    }

    static std::vector<PersistentStopEventId> collectActiveEvents(const Data& data, const PersistentTripId tripId) {
        const PersistentTrip& trip = data.trips_[tripId];
        std::vector<PersistentStopEventId> events;
        events.reserve(trip.numberOfEvents);

        const auto first = static_cast<std::size_t>(trip.firstEvent);
        for (std::uint32_t i = 0; i < trip.numberOfEvents; ++i) {
            const PersistentStopEventId eventId(first + i);
            if (data.events_[eventId].isSkipped) continue;
            events.push_back(eventId);
        }

        return events;
    }

    static std::pair<PersistentRouteId, std::uint32_t> findCompatibleRoute(
        Data& data, const std::vector<StopId>& stopSequence, std::span<const PersistentStopEvent> events) {
        const std::size_t h = data.hashStopSequence(stopSequence);
        auto it = data.routesBySequenceHash_.find(h);

        if (it != data.routesBySequenceHash_.end()) {
            for (const PersistentRouteId candidate : it->second) {
                if (!data.isRoute(candidate)) continue;
                if (data.stopSequenceEquals(data.routes_[candidate].stopSequence, stopSequence)) {
                    if (auto safeIdx = findFiFoCompatibleIndex(data, candidate, events)) {
                        return {candidate, *safeIdx};
                    }
                }
            }
        }

        const PersistentRouteId newId(data.routes_.size());
        PersistentRoute r;
        r.routeId = newId;
        r.stopSequence = stopSequence;

        data.routes_.push_back(std::move(r));
        data.routesBySequenceHash_[h].push_back(newId);

        return {newId, 0};
    }

    static Time getFirstDepartureTime(const Data& data, const PersistentTripId tripId) {
        const PersistentTrip& trip = data.trips_[tripId];
        for (std::uint32_t i = 0; i < trip.numberOfEvents; i++) {
            const PersistentStopEvent& e = data.events_[trip.firstEvent + i];
            if (!e.isSkipped) return e.departureTime;
        }
        // Should not be reachable for structurally sound trips, but provide a fallback
        return Time(std::numeric_limits<int>::max());
    }

    static std::vector<StopId> getEffectiveSequence(const Data& data, const PersistentTripId tripId) {
        std::vector<StopId> seq;
        const PersistentTrip& trip = data.trips_[tripId];
        seq.reserve(trip.numberOfEvents);
        for (std::uint32_t i = 0; i < trip.numberOfEvents; i++) {
            const PersistentStopEvent& e = data.events_[trip.firstEvent + i];
            if (!e.isSkipped) seq.push_back(e.stop);
        }
        return seq;
    }

    // Returns:
    // -1 : eventsA is safely BEFORE eventsB (or exactly equal)
    //  1 : eventsA is safely AFTER eventsB
    //  0 : FIFO Violation (crossover detected)
    static int compareFifo(const std::span<const PersistentStopEvent> eventsA,
                           const std::span<const PersistentStopEvent> eventsB) {
        bool a_strictly_before = false;
        bool b_strictly_before = false;

        std::uint32_t ia = 0, ib = 0;
        while (ia < eventsA.size() && ib < eventsB.size()) {
            const PersistentStopEvent& ea = eventsA[ia];
            if (ea.isSkipped) {
                ia++;
                continue;
            }
            const PersistentStopEvent& eb = eventsB[ib];
            if (eb.isSkipped) {
                ib++;
                continue;
            }

            // Check if A is faster/earlier
            if (ea.arrivalTime < eb.arrivalTime || ea.departureTime < eb.departureTime) {
                a_strictly_before = true;
            }
            // Check if B is faster/earlier
            if (ea.arrivalTime > eb.arrivalTime || ea.departureTime > eb.departureTime) {
                b_strictly_before = true;
            }

            // If lines cross at any point, it's a structural violation
            if (a_strictly_before && b_strictly_before) {
                return 0;
            }

            ia++;
            ib++;
        }

        // No crossovers detected. Determine the overall direction.
        if (b_strictly_before) return 1;  // A comes AFTER B

        return -1;  // A comes BEFORE B (or they are identical, defaulting to stable insertion)
    }

    // Wrapper Overloads
    static int compareFifo(const Data& data, const PersistentTripId a, const PersistentTripId b) {
        return compareFifo(getTripEvents(data, a), getTripEvents(data, b));
    }

    static int compareFifo(const Data& data, const PersistentTripId a, std::span<const PersistentStopEvent> b_events) {
        return compareFifo(getTripEvents(data, a), b_events);
    }

    static int compareFifo(const Data& data, std::span<const PersistentStopEvent> a_events, const PersistentTripId b) {
        return compareFifo(a_events, getTripEvents(data, b));
    }

    static std::optional<std::uint32_t> findFiFoCompatibleIndex(const Data& data, const PersistentRouteId routeId,
                                                                std::span<const PersistentStopEvent> events) {
        const auto& list = data.routes_[routeId].trips;
        if (list.empty()) return 0;

        std::int32_t low = 0;
        auto high = static_cast<std::int32_t>(list.size());

        while (low < high) {
            std::int32_t mid = low + (high - low) / 2;
            int cmp = compareFifo(data, events, list[mid]);

            if (cmp == 0) {
                // Direct crossover detected during traversal
                return std::nullopt;
            } else if (cmp < 0) {
                // events come BEFORE list[mid], insertion must be at or before mid
                high = mid;
            } else {
                // events come AFTER list[mid], insertion must be after mid
                low = mid + 1;
            }
        }
        auto insertIdx = static_cast<std::uint32_t>(low);
        return insertIdx;
    }

    static void finalizeChangeSummery(const Data& data, UpdateContext& context) {
        auto& incomingTracking = context.summary.tripsToRediscoverIncomingDueToCancellation;
        for (const auto& [routeId, tripId] : context.tripsAfterExtractedTrips) {
            if (const auto& trip = data.trips_[tripId]; trip.isActive && trip.route == routeId) {
                context.summary.tripsToRediscoverIncomingDueToCancellation.push_back(tripId);
            }
        }
        std::ranges::sort(incomingTracking);

    }
};

}  // namespace DynamicTimeTable::Algo
