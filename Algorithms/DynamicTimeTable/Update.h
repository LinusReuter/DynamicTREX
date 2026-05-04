#pragma once

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../DataStructures/DynamicTimeTable/Data.h"
#include "../../DataStructures/DynamicTimeTable/UpdateTypes.h"

namespace DynamicTimeTable {
namespace Algo {

struct UpdateContext {
    ChangeSummary summary;
    std::vector<PersistentTripId> extractionQueue;
    // Group modified trips by route to optimize pairwise FIFO checks
    std::unordered_map<PersistentRouteId, std::vector<PersistentTripId>> modifiedTripsByRoute;
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

        data.latestChanges_ = std::move(context.summary);

        return stats;
    }

private:
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
            trip.isActive = false;

            if (data.isRoute(oldRoute)) {
                auto& list = data.routes_[oldRoute].trips;
                // Preserve chronological order of remaining trips
                list.erase(std::remove(list.begin(), list.end(), tripId), list.end());
            }

            context.summary.cancelledTrips.push_back({tripId, oldRoute});
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

            for (const StopModification& m : mods) {
                const std::size_t idx = static_cast<std::size_t>(m.stopIndex);
                if (idx >= trip.numberOfEvents) {
                    structural = true;
                    continue;
                }

                const std::size_t first = static_cast<std::size_t>(trip.firstEvent);
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

                // #TODO currently only allows cancellation not reactivation of events
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
                extractTrip(data, tripId, context);
            } else {
                context.modifiedTripsByRoute[trip.route].push_back(tripId);
                stats.successfulUpdates++;
            }
        }

        // Restore chronological sort order for affected routes.
        // NOTE: std::stable_sort is robust and easy. However, if no extractions occur and only
        // small time changes happen, local swaps (e.g. insertion sort) would likely be more efficient.
        for (const auto& [routeId, modTrips] : context.modifiedTripsByRoute) {
            auto& list = data.routes_[routeId].trips;
            std::stable_sort(list.begin(), list.end(), [&data](PersistentTripId a, PersistentTripId b) {
                return getFirstDepartureTime(data, a) < getFirstDepartureTime(data, b);
            });
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
                auto it = std::find(list.begin(), list.end(), m_id);
                if (it == list.end()) continue;
                std::ptrdiff_t idx = std::distance(list.begin(), it);

                // Check backwards
                for (std::ptrdiff_t i = idx - 1; i >= 0; --i) {
                    if (checkFifoViolation(data, list[i], m_id)) {
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
                    if (checkFifoViolation(data, m_id, list[i])) {
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

                extractTrip(data, worstTrip, context);

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

            PersistentRouteId routeId = findCompatibleRoute(data, add.stopSequence, newTripId);
            trip.route = routeId;
            data.trips_.push_back(trip);

            insertTripChronologically(data, routeId, newTripId);
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

            PersistentRouteId routeId = findCompatibleRoute(data, effectiveSeq, tripId);
            trip.route = routeId;
            insertTripChronologically(data, routeId, tripId);
            context.summary.addedTrips.push_back(tripId);
        }

        return stats;
    }

    static void extractTrip(Data& data, const PersistentTripId tripId, UpdateContext& context) {
        PersistentTrip& trip = data.trips_[tripId];
        const PersistentRouteId oldRoute = trip.route;

        if (data.isRoute(oldRoute)) {
            auto& list = data.routes_[oldRoute].trips;
            list.erase(std::remove(list.begin(), list.end(), tripId), list.end());
        }

        context.summary.cancelledTrips.push_back({tripId, oldRoute});
        context.extractionQueue.push_back(tripId);
    }

    static PersistentRouteId findCompatibleRoute(Data& data, const std::vector<StopId>& stopSequence,
                                                 const PersistentTripId tripId) {
        const std::size_t h = data.hashStopSequence(stopSequence);
        auto it = data.routesBySequenceHash_.find(h);

        if (it != data.routesBySequenceHash_.end()) {
            for (const PersistentRouteId candidate : it->second) {
                if (!data.isRoute(candidate)) continue;
                if (data.stopSequenceEquals(data.routes_[candidate].stopSequence, stopSequence)) {
                    if (isFifoCompatible(data, candidate, tripId)) {
                        return candidate;
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

        return newId;
    }

    static bool isFifoCompatible(const Data& data, const PersistentRouteId routeId, const PersistentTripId tripId) {
        const auto& list = data.routes_[routeId].trips;
        if (list.empty()) return true;

        Time dep = getFirstDepartureTime(data, tripId);
        auto it = std::upper_bound(list.begin(), list.end(), dep,
                                   [&data](Time val, PersistentTripId t) { return val < getFirstDepartureTime(data, t); });

        // Check predecessor
        if (it != list.begin()) {
            PersistentTripId pred = *(it - 1);
            if (checkFifoViolation(data, pred, tripId)) return false;
        }
        // Check successor
        if (it != list.end()) {
            PersistentTripId succ = *it;
            if (checkFifoViolation(data, tripId, succ)) return false;
        }

        return true;
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

    static void insertTripChronologically(Data& data, const PersistentRouteId routeId, const PersistentTripId tripId) {
        auto& list = data.routes_[routeId].trips;
        Time dep = getFirstDepartureTime(data, tripId);
        auto it = std::upper_bound(list.begin(), list.end(), dep,
                                   [&data](Time val, PersistentTripId t) { return val < getFirstDepartureTime(data, t); });
        list.insert(it, tripId);
    }

    static bool checkFifoViolation(const Data& data, const PersistentTripId a, const PersistentTripId b) {
        // Assume 'a' is scheduled to run chronologically before 'b'.
        const PersistentTrip& ta = data.trips_[a];
        const PersistentTrip& tb = data.trips_[b];

        std::uint32_t ia = 0, ib = 0;
        while (ia < ta.numberOfEvents && ib < tb.numberOfEvents) {
            const PersistentStopEvent& ea = data.events_[ta.firstEvent + ia];
            if (ea.isSkipped) {
                ia++;
                continue;
            }
            const PersistentStopEvent& eb = data.events_[tb.firstEvent + ib];
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
};

}  // namespace Algo
}  // namespace DynamicTimeTable
