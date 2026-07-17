#pragma once

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string_view>

#include "String/String.h"

// Structure-level memory footprint of the persistent (mid/long-term) data
// structures maintained across incremental updates. All figures are computed
// from live container sizes/capacities (no OS/RSS involvement), so they
// attribute memory precisely to each structure and are portable.
//
// "Logical" = size() * sizeof(element) (the data actually in use).
// "Capacity" = capacity() * sizeof(element) + container overhead (what is
// actually reserved). The gap between them is the reallocation/fragmentation
// slack that the sequential-update locality analysis cares about.
struct TransferStoreMemory {
    std::size_t nodes = 0;
    std::size_t outEdges = 0;
    std::size_t inEdges = 0;
    // Edge payload: the (target, meta) / source-id arrays themselves.
    long long outPayloadLogical = 0;
    long long outPayloadReserved = 0;
    long long inPayloadLogical = 0;
    long long inPayloadReserved = 0;
    // Management structures that are not edge payload: the per-node vector
    // control blocks (one std::vector header per node, both directions) plus the
    // two top-level node-vector objects and their slack.
    long long nodeHeaderBytes = 0;
    // Stripe spinlocks (fixed, cache-line aligned).
    long long lockBytes = 0;

    long long managementBytes() const noexcept { return nodeHeaderBytes + lockBytes; }
    long long payloadLogicalBytes() const noexcept { return outPayloadLogical + inPayloadLogical; }
    long long payloadReservedBytes() const noexcept { return outPayloadReserved + inPayloadReserved; }
    // Store top-level footprint: edge payload + management structures.
    long long totalLogicalBytes() const noexcept { return payloadLogicalBytes() + managementBytes(); }
    long long totalReservedBytes() const noexcept { return payloadReservedBytes() + managementBytes(); }
    double overheadRatio() const noexcept {
        return payloadLogicalBytes() > 0
                   ? static_cast<double>(totalReservedBytes()) / static_cast<double>(payloadLogicalBytes())
                   : 0.0;
    }
};

struct UpdateMemoryStats {
    TransferStoreMemory store;
    // DynamicQueryData: translation tables + nested TripBased::QueryData.
    long long queryDataLogicalBytes = 0;
    long long queryDataCapacityBytes = 0;
    // DynamicTimeTable::Data backing storage.
    long long timeTableLogicalBytes = 0;
    long long timeTableCapacityBytes = 0;

    long long totalLogicalBytes() const noexcept {
        return store.totalLogicalBytes() + queryDataLogicalBytes + timeTableLogicalBytes;
    }
    long long totalCapacityBytes() const noexcept {
        return store.totalReservedBytes() + queryDataCapacityBytes + timeTableCapacityBytes;
    }
};

inline void printMemoryStats(const UpdateMemoryStats& m, std::ostream& out = std::cout) {
    auto row = [&](std::string_view name, long long logical, long long capacity) {
        out << "  " << std::left << std::setw(26) << name << std::right << std::setw(14)
            << String::bytesToString(logical) << " logical  " << std::setw(14) << String::bytesToString(capacity)
            << " reserved\n";
    };
    out << "Memory footprint (structure byteSize)\n";
    out << "  TransferStore nodes=" << m.store.nodes << " outEdges=" << m.store.outEdges
        << " inEdges=" << m.store.inEdges << "\n";
    row("  store edges (out)", m.store.outPayloadLogical, m.store.outPayloadReserved);
    row("  store edges (in)", m.store.inPayloadLogical, m.store.inPayloadReserved);
    row("  store management", m.store.managementBytes(), m.store.managementBytes());
    row("  store TOTAL", m.store.totalLogicalBytes(), m.store.totalReservedBytes());
    row("query data", m.queryDataLogicalBytes, m.queryDataCapacityBytes);
    row("dynamic timetable", m.timeTableLogicalBytes, m.timeTableCapacityBytes);
    row("TOTAL", m.totalLogicalBytes(), m.totalCapacityBytes());
    out << "  store over-capacity ratio: " << std::fixed << std::setprecision(3) << m.store.overheadRatio() << "\n";
}
