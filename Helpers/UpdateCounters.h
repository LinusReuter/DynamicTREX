#pragma once

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string_view>

// Work-quantity counters for one incremental transfer update. Populated only in
// "detail" builds (see DYN_COLLECT_TRANSFER_STATS / collectTransferStats in
// Algorithms/DynamicTB/preprocessing/TransferUpdate.h); in normal builds every
// increment is guarded by `if constexpr (collectTransferStats)` and compiles
// away, so this struct stays all-zero and the hot path is untouched.
//
// Counters are grouped by the phase that produces them. Query-data-rebuild
// figures (active*/skipped) are derived from result sizes, not instrumented.
struct TransferUpdateCounters {
    // --- Cancellation phase (TransferUpdate Phase 1) ---
    std::uint64_t cancelledTripsProcessed = 0;
    std::uint64_t outgoingEdgesCleared = 0;
    std::uint64_t incomingEdgesCleared = 0;

    // --- Discovery targets ---
    std::uint64_t discoverOutgoingEvents = 0;
    std::uint64_t discoverIncomingEvents = 0;

    // --- Outgoing discovery (Phase 2) ---
    std::uint64_t outgoingEdgesDiscovered = 0;
    std::uint64_t outgoingEdgesAdded = 0;
    std::uint64_t outgoingEdgesRemoved = 0;

    // --- Incoming discovery (Phase 3) ---
    std::uint64_t incomingEdgesDiscovered = 0;
    std::uint64_t incomingEdgesAdded = 0;
    std::uint64_t incomingEdgesRemoved = 0;
    std::uint64_t dominationCleanups = 0;
    std::uint64_t dominationEdgesRemoved = 0;

    // --- Changed-arrival propagation (Phase 4) ---
    std::uint64_t arrivalPropagationTrips = 0;
    std::uint64_t arrivalUpstreamSources = 0;

    // --- Minimization ---
    std::uint64_t tripsMinimized = 0;
    std::uint64_t minStopsScanned = 0;
    std::uint64_t minCandidatesEvaluated = 0;
    std::uint64_t minCandidatesKept = 0;
    std::uint64_t minWarmStartReplays = 0;
    // Edges whose isMinimized flag flipped this update; the seed of the TREX level-0
    // affected set (see AffectedEventSink.h).
    std::uint64_t minimizationFlips = 0;

    // --- TREX customization ---
    // Distinct stop events in the merged level-0 affected set.
    std::uint64_t affectedEventsLevel0 = 0;

    // --- Query-data rebuild (derived from result sizes) ---
    std::uint64_t activeRoutes = 0;
    std::uint64_t activeTrips = 0;
    std::uint64_t activeEvents = 0;
    std::uint64_t skippedEvents = 0;

    TransferUpdateCounters& operator+=(const TransferUpdateCounters& o) noexcept {
        cancelledTripsProcessed += o.cancelledTripsProcessed;
        outgoingEdgesCleared += o.outgoingEdgesCleared;
        incomingEdgesCleared += o.incomingEdgesCleared;
        discoverOutgoingEvents += o.discoverOutgoingEvents;
        discoverIncomingEvents += o.discoverIncomingEvents;
        outgoingEdgesDiscovered += o.outgoingEdgesDiscovered;
        outgoingEdgesAdded += o.outgoingEdgesAdded;
        outgoingEdgesRemoved += o.outgoingEdgesRemoved;
        incomingEdgesDiscovered += o.incomingEdgesDiscovered;
        incomingEdgesAdded += o.incomingEdgesAdded;
        incomingEdgesRemoved += o.incomingEdgesRemoved;
        dominationCleanups += o.dominationCleanups;
        dominationEdgesRemoved += o.dominationEdgesRemoved;
        arrivalPropagationTrips += o.arrivalPropagationTrips;
        arrivalUpstreamSources += o.arrivalUpstreamSources;
        tripsMinimized += o.tripsMinimized;
        minStopsScanned += o.minStopsScanned;
        minCandidatesEvaluated += o.minCandidatesEvaluated;
        minCandidatesKept += o.minCandidatesKept;
        minWarmStartReplays += o.minWarmStartReplays;
        minimizationFlips += o.minimizationFlips;
        affectedEventsLevel0 += o.affectedEventsLevel0;
        activeRoutes += o.activeRoutes;
        activeTrips += o.activeTrips;
        activeEvents += o.activeEvents;
        skippedEvents += o.skippedEvents;
        return *this;
    }
};

// Accumulates counters across many timeline steps / iterations (sum + count),
// so an end-of-run summary can report totals and per-step means.
class TransferCountersAccumulator {
public:
    void add(const TransferUpdateCounters& sample) noexcept {
        sum += sample;
        ++count;
    }
    std::size_t sampleCount() const noexcept { return count; }
    const TransferUpdateCounters& sumCounters() const noexcept { return sum; }

private:
    TransferUpdateCounters sum{};
    std::size_t count{0};
};

inline void printTransferUpdateCounters(const TransferUpdateCounters& c, std::ostream& out = std::cout) {
    auto row = [&](std::string_view name, std::uint64_t value) {
        out << "  " << std::left << std::setw(28) << name << std::right << std::setw(14) << value << "\n";
    };
    out << "Transfer update counters\n";
    out << " Cancellation:\n";
    row("cancelled trips", c.cancelledTripsProcessed);
    row("outgoing edges cleared", c.outgoingEdgesCleared);
    row("incoming edges cleared", c.incomingEdgesCleared);
    out << " Discovery:\n";
    row("outgoing target events", c.discoverOutgoingEvents);
    row("incoming target events", c.discoverIncomingEvents);
    row("outgoing edges discovered", c.outgoingEdgesDiscovered);
    row("outgoing edges added", c.outgoingEdgesAdded);
    row("outgoing edges removed", c.outgoingEdgesRemoved);
    row("incoming edges discovered", c.incomingEdgesDiscovered);
    row("incoming edges added", c.incomingEdgesAdded);
    row("incoming edges removed", c.incomingEdgesRemoved);
    row("domination cleanups", c.dominationCleanups);
    row("domination edges removed", c.dominationEdgesRemoved);
    out << " Arrival propagation:\n";
    row("changed-arrival trips", c.arrivalPropagationTrips);
    row("upstream sources flagged", c.arrivalUpstreamSources);
    out << " Minimization:\n";
    row("trips minimized", c.tripsMinimized);
    row("stops scanned", c.minStopsScanned);
    row("candidates evaluated", c.minCandidatesEvaluated);
    row("candidates kept", c.minCandidatesKept);
    row("warm-start replays", c.minWarmStartReplays);
    row("isMinimized flips", c.minimizationFlips);
    out << " TREX customization:\n";
    row("level-0 affected events", c.affectedEventsLevel0);
    out << " Query data:\n";
    row("active routes", c.activeRoutes);
    row("active trips", c.activeTrips);
    row("active events", c.activeEvents);
    row("skipped events", c.skippedEvents);
}
