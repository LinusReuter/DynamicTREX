#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../DataStructures/DynamicTimeTable/Data.h"
#include "../../DataStructures/DynamicTimeTable/UpdateTypes.h"

namespace DynamicTimeTable::Algo {

// Simple, deterministic update simulator for testing.
// Produces PendingUpdates based on current time and a seeded RNG.

enum class FutureStopPolicy { NextStop, RandomFutureStop };

struct UpdateSimulationConfig {
    uint32_t seed = 42;

    struct CancellationConfig {
        std::size_t expectedCount = 0;
        Time horizon = Time(1 * 3600);       // consider trips starting within this horizon
        Time minLeadTime = Time(5 * 60);     // avoid cancelling trips that start too soon
    } cancellations;

    struct DelayConfig {
        std::size_t expectedCount = 0;
        Time minInitialDelay = Time(60);
        Time maxInitialDelay = Time(30 * 60);
        double travelTimeVariation = 0.2;  // +/- fraction of leg travel time
        FutureStopPolicy stopPolicy = FutureStopPolicy::RandomFutureStop;
    } delays;

    struct SkipConfig {
        std::size_t expectedCount = 0;
        Time horizon = Time(2 * 3600);      // consider trips starting within this horizon
        bool includeInProgress = true;
        bool excludeFirstAndLastStop = true;
        int maxSkippedStopsPerTrip = 1;
        FutureStopPolicy stopPolicy = FutureStopPolicy::RandomFutureStop;
    } skips;
};

struct UpdateSimulationStats {
    std::size_t cancelledTrips = 0;
    std::size_t delayedTrips = 0;
    std::size_t skippedTrips = 0;
    std::size_t modifiedStops = 0;
};

class UpdateSimulator {
public:
    explicit UpdateSimulator(UpdateSimulationConfig cfg = {}) : cfg_(std::move(cfg)), rng_(cfg_.seed) {}

    void reseed(const uint32_t seed) {
        cfg_.seed = seed;
        rng_.seed(seed);
    }

    PendingUpdates generate(const Data& data, const Time now, UpdateSimulationStats* outStats = nullptr) {
        UpdateSimulationStats stats{};
        PendingUpdates result;

        const auto& trips = data.trips();
        const auto& events = data.events();

        std::vector<Time> firstDeparture(trips.size(), noTime);
        std::vector<Time> lastArrival(trips.size(), noTime);

        // Precompute basic time windows per trip
        for (std::size_t i = 0; i < trips.size(); i++) {
            const PersistentTrip& trip = trips[i];
            if (!trip.isActive) continue;

            Time firstDep = noTime;
            Time lastArr = noTime;

            for (std::uint32_t j = 0; j < trip.numberOfEvents; j++) {
                const PersistentStopEvent& e = events[trip.firstEvent + j];
                if (e.isSkipped) continue;

                if (firstDep == noTime && e.departureTime != noTime) {
                    firstDep = e.departureTime;
                }
                if (e.arrivalTime != noTime) {
                    lastArr = e.arrivalTime;
                } else if (e.departureTime != noTime) {
                    lastArr = e.departureTime;
                }
            }

            firstDeparture[i] = firstDep;
            lastArrival[i] = lastArr;
        }

        std::vector<PersistentTripId> cancellationCandidates;
        std::vector<PersistentTripId> delayCandidates;
        std::vector<PersistentTripId> skipCandidates;

        // Build candidate sets
        for (std::size_t i = 0; i < trips.size(); i++) {
            const PersistentTrip& trip = trips[i];
            if (!trip.isActive) continue;

            const Time firstDep = firstDeparture[i];
            const Time lastArr = lastArrival[i];
            if (firstDep == noTime || lastArr == noTime) continue;

            const bool futureTrip = firstDep > now;
            const bool inProgress = (firstDep <= now) && (now < lastArr);

            if (futureTrip) {
                if (firstDep <= now + cfg_.cancellations.horizon &&
                    firstDep >= now + cfg_.cancellations.minLeadTime) {
                    cancellationCandidates.emplace_back(PersistentTripId(i));
                }
                if (firstDep <= now + cfg_.skips.horizon) {
                    skipCandidates.emplace_back(PersistentTripId(i));
                }
            }

            if (inProgress) {
                delayCandidates.emplace_back(PersistentTripId(i));
                if (cfg_.skips.includeInProgress) {
                    skipCandidates.emplace_back(PersistentTripId(i));
                }
            }
        }

        // Select cancellations
        const std::vector<PersistentTripId> cancellations = sampleSubset(cancellationCandidates, cfg_.cancellations.expectedCount);
        std::unordered_set<PersistentTripId> cancelledSet(cancellations.begin(), cancellations.end());

        result.cancellations = cancellations;
        std::sort(result.cancellations.begin(), result.cancellations.end());
        stats.cancelledTrips = result.cancellations.size();

        // Filter delay/skip candidates against cancellations
        std::vector<PersistentTripId> filteredDelayCandidates;
        filteredDelayCandidates.reserve(delayCandidates.size());
        for (const PersistentTripId id : delayCandidates) {
            if (cancelledSet.count(id) == 0) filteredDelayCandidates.push_back(id);
        }

        std::vector<PersistentTripId> filteredSkipCandidates;
        filteredSkipCandidates.reserve(skipCandidates.size());
        for (const PersistentTripId id : skipCandidates) {
            if (cancelledSet.count(id) == 0) filteredSkipCandidates.push_back(id);
        }

        const std::vector<PersistentTripId> selectedDelays = sampleSubset(filteredDelayCandidates, cfg_.delays.expectedCount);
        const std::vector<PersistentTripId> selectedSkips = sampleSubset(filteredSkipCandidates, cfg_.skips.expectedCount);

        std::vector<TripModificationBuffer> modBuffers(trips.size());

        // Apply delays
        for (const PersistentTripId tripId : selectedDelays) {
            const std::size_t tIdx = static_cast<std::size_t>(tripId);
            if (tIdx >= trips.size()) continue;
            const PersistentTrip& trip = trips[tIdx];

            const std::vector<std::size_t> futureStops = collectFutureStops(data, trip, now, true, false);
            if (futureStops.empty()) continue;

            const std::size_t startIdx = chooseStopIndex(futureStops, cfg_.delays.stopPolicy);
            const Time delay = sampleTime(cfg_.delays.minInitialDelay, cfg_.delays.maxInitialDelay);

            if (delay == noTime || delay == Time(0)) continue;

            const bool applied = applyDelayToTrip(data, tripId, startIdx, delay, modBuffers[tIdx]);
            if (applied) {
                stats.delayedTrips++;
            }
        }

        // Apply skipped stops
        for (const PersistentTripId tripId : selectedSkips) {
            const std::size_t tIdx = static_cast<std::size_t>(tripId);
            if (tIdx >= trips.size()) continue;
            const PersistentTrip& trip = trips[tIdx];

            std::vector<std::size_t> futureStops = collectFutureStops(data, trip, now, true, cfg_.skips.excludeFirstAndLastStop);
            if (futureStops.empty()) continue;

            int maxSkip = cfg_.skips.maxSkippedStopsPerTrip;
            if (maxSkip <= 0) continue;
            if (static_cast<std::size_t>(maxSkip) > futureStops.size()) {
                maxSkip = static_cast<int>(futureStops.size());
            }
            if (maxSkip <= 0) continue;

            const int numToSkip = sampleInt(1, maxSkip);
            std::shuffle(futureStops.begin(), futureStops.end(), rng_);
            futureStops.resize(static_cast<std::size_t>(numToSkip));

            for (const std::size_t stopIdx : futureStops) {
                modBuffers[tIdx].addSkip(stopIdx);
                stats.modifiedStops++;
            }
            stats.skippedTrips++;
        }

        // Emit modifications (sorted by trip id, stop index)
        for (std::size_t i = 0; i < modBuffers.size(); i++) {
            if (!modBuffers[i].used()) continue;
            auto mods = modBuffers[i].takeSorted();
            if (!mods.empty()) {
                result.modifications.emplace_back(PersistentTripId(i), std::move(mods));
            }
        }

        if (outStats) *outStats = stats;
        return result;
    }

private:
    struct TripModificationBuffer {
        std::vector<StopModification> modifications;
        std::unordered_map<std::size_t, std::size_t> indexToPos;
        bool active = false;

        bool used() const { return active && !modifications.empty(); }

        void addStopModification(const std::size_t stopIdx, const Time newArrival, const Time newDeparture, const bool skipped) {
            active = true;
            const auto it = indexToPos.find(stopIdx);
            if (it == indexToPos.end()) {
                StopModification mod;
                mod.stopIndex = StopIndex(stopIdx);
                mod.newArrivalTime = newArrival;
                mod.newDepartureTime = newDeparture;
                mod.isSkipped = skipped;
                indexToPos[stopIdx] = modifications.size();
                modifications.push_back(mod);
            } else {
                StopModification& mod = modifications[it->second];
                if (newArrival != noTime) mod.newArrivalTime = newArrival;
                if (newDeparture != noTime) mod.newDepartureTime = newDeparture;
                if (skipped) mod.isSkipped = true;
            }
        }

        void addSkip(const std::size_t stopIdx) { addStopModification(stopIdx, noTime, noTime, true); }

        std::vector<StopModification> takeSorted() {
            std::vector<StopModification> out = std::move(modifications);
            std::sort(out.begin(), out.end(), [](const StopModification& a, const StopModification& b) {
                return a.stopIndex < b.stopIndex;
            });
            modifications.clear();
            indexToPos.clear();
            active = false;
            return out;
        }
    };

    static inline int64_t toInt(const Time t) { return static_cast<int64_t>(t.value()); }

    static inline Time clampToTime(const int64_t value) {
        int64_t v = value;
        if (v < 0) v = 0;
        const int64_t maxValid = static_cast<int64_t>(Time::InvalidValue) - 1;
        if (v > maxValid) v = maxValid;
        return Time(static_cast<Time::ValueType>(v));
    }

    int sampleInt(const int minVal, const int maxVal) {
        if (minVal >= maxVal) return minVal;
        std::uniform_int_distribution<int> dist(minVal, maxVal);
        return dist(rng_);
    }

    Time sampleTime(Time minVal, Time maxVal) {
        const auto minV = static_cast<Time::ValueType>(minVal.value());
        const auto maxV = static_cast<Time::ValueType>(maxVal.value());
        const auto lo = std::min(minV, maxV);
        const auto hi = std::max(minV, maxV);
        std::uniform_int_distribution<Time::ValueType> dist(lo, hi);
        return Time(dist(rng_));
    }

    std::size_t chooseStopIndex(const std::vector<std::size_t>& indices, const FutureStopPolicy policy) {
        if (indices.empty()) return 0;
        if (policy == FutureStopPolicy::NextStop) return indices.front();
        const int idx = sampleInt(0, static_cast<int>(indices.size() - 1));
        return indices[static_cast<std::size_t>(idx)];
    }

    std::vector<std::size_t> collectFutureStops(const Data& data, const PersistentTrip& trip, const Time now,
                                                const bool excludeSkipped, const bool excludeFirstLast) const {
        std::vector<std::size_t> indices;
        indices.reserve(trip.numberOfEvents);

        for (std::uint32_t i = 0; i < trip.numberOfEvents; i++) {
            const PersistentStopEvent& e = data.events()[trip.firstEvent + i];
            if (excludeSkipped && e.isSkipped) continue;

            Time eventTime = noTime;
            if (e.arrivalTime != noTime) {
                eventTime = e.arrivalTime;
            } else if (e.departureTime != noTime) {
                eventTime = e.departureTime;
            }

            if (eventTime == noTime) continue;
            if (eventTime <= now) continue;

            if (excludeFirstLast) {
                if (i == 0 || i + 1 == trip.numberOfEvents) continue;
            }

            indices.push_back(i);
        }

        return indices;
    }

    bool applyDelayToTrip(const Data& data, const PersistentTripId tripId, const std::size_t startIdx, const Time initialDelay,
                          TripModificationBuffer& buffer) {
        const auto& trips = data.trips();
        const auto& events = data.events();
        const std::size_t tIdx = static_cast<std::size_t>(tripId);
        if (tIdx >= trips.size()) return false;

        const PersistentTrip& trip = trips[tIdx];
        if (startIdx >= trip.numberOfEvents) return false;

        int64_t currentDelay = toInt(initialDelay);
        if (currentDelay <= 0) return false;

        for (std::size_t i = startIdx; i < trip.numberOfEvents; i++) {
            const PersistentStopEvent& e = events[trip.firstEvent + i];

            if (e.arrivalTime != noTime) {
                buffer.addStopModification(i, clampToTime(toInt(e.arrivalTime) + currentDelay), noTime, false);
            }
            if (e.departureTime != noTime) {
                buffer.addStopModification(i, noTime, clampToTime(toInt(e.departureTime) + currentDelay), false);
            }

            if (i + 1 < trip.numberOfEvents) {
                const PersistentStopEvent& next = events[trip.firstEvent + i + 1];
                if (e.departureTime != noTime && next.arrivalTime != noTime) {
                    int64_t travelTime = toInt(next.arrivalTime) - toInt(e.departureTime);
                    if (travelTime < 0) travelTime = 0;

                    const double var = static_cast<double>(travelTime) * cfg_.delays.travelTimeVariation;
                    const int64_t delta = static_cast<int64_t>(std::llround(var));
                    if (delta > 0) {
                        std::uniform_int_distribution<int64_t> dist(-delta, delta);
                        currentDelay += dist(rng_);
                        if (currentDelay < 0) currentDelay = 0;
                    }
                }
            }
        }

        return true;
    }

    std::vector<PersistentTripId> sampleSubset(std::vector<PersistentTripId> candidates, const std::size_t count) {
        if (candidates.empty()) return {};
        std::shuffle(candidates.begin(), candidates.end(), rng_);
        if (candidates.size() > count) candidates.resize(count);
        std::sort(candidates.begin(), candidates.end());
        return candidates;
    }

    UpdateSimulationConfig cfg_;
    std::mt19937 rng_;
};

}  // namespace DynamicTimeTable::Algo
