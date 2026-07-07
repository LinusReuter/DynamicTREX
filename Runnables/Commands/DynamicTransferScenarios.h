#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../Algorithms/DynamicTimeTable/BuildQueryData.h"
#include "../../Algorithms/DynamicTimeTable/Update.h"
#include "../../Algorithms/DynamicTimeTable/UpdateSimulation.h"
#include "../../DataStructures/DynamicTimeTable/Data.h"
#include "../../DataStructures/TransferStore/TransferStore.h"
#include "Algorithms/DynamicTB/preprocessing/TransferUpdate.h"

namespace DynamicTransferScenarios {

using DynamicQueryData = DynamicTimeTable::Algo::DynamicQueryData;
using TransferMeta = DynamicTB::Preprocessing::TransferMeta;
using TransferStoreType = TransferStore<PersistentStopEventId, TransferMeta>;
using TransferUpdater = DynamicTB::Preprocessing::TransferUpdate;

enum class TransferSetKind { Full, Reduced };

enum class TransferSetSelection { Full, Reduced, Both };

struct TransferSetConfig {
    int threads{1};
    TransferSetSelection selection{TransferSetSelection::Both};
    std::vector<TransferSetKind> kinds{TransferSetKind::Full, TransferSetKind::Reduced};
};

struct ComparisonResult {
    bool transferValidation{false};
    bool matchesRebuild{false};
    std::size_t incrementalEdges{0};
    std::size_t rebuiltEdges{0};
};

struct AppliedUpdate {
    AppliedUpdate(DynamicQueryData queryData,
                  const DynamicTimeTable::ChangeSummary& changes,
                  const DynamicTimeTable::UpdateStatistics& statistics)
        : queryData(std::move(queryData)), changes(&changes), statistics(statistics) {}

    DynamicQueryData queryData;
    const DynamicTimeTable::ChangeSummary* changes{nullptr};
    DynamicTimeTable::UpdateStatistics statistics{};
};

struct TimedAppliedUpdate {
    AppliedUpdate applied;
    std::chrono::microseconds duration{};
};

struct UpdateComparisonResult {
    AppliedUpdate applied;
    DynamicTimeTable::PendingUpdates pendingUpdates;
    DynamicTimeTable::Algo::UpdateSimulationStats simulationStats{};
    bool ok{false};
};

template <typename Value>
struct TimedResult {
    Value value;
    std::chrono::microseconds duration{};
};

template <typename Function>
inline std::chrono::microseconds timeAction(Function&& function) {
    const auto start = std::chrono::high_resolution_clock::now();
    function();
    const auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
}

template <typename Function>
inline auto runTimed(Function&& function) {
    using Value = decltype(function());
    const auto start = std::chrono::high_resolution_clock::now();
    Value value = function();
    const auto stop = std::chrono::high_resolution_clock::now();
    return TimedResult<Value>{std::move(value), std::chrono::duration_cast<std::chrono::microseconds>(stop - start)};
}

inline std::string_view nameOf(const TransferSetKind kind) noexcept {
    return kind == TransferSetKind::Full ? "full" : "reduced";
}

inline std::vector<TransferSetKind> expandTransferSetSelection(const TransferSetSelection selection) {
    switch (selection) {
        case TransferSetSelection::Full:
            return {TransferSetKind::Full};
        case TransferSetSelection::Reduced:
            return {TransferSetKind::Reduced};
        case TransferSetSelection::Both:
            return {TransferSetKind::Full, TransferSetKind::Reduced};
    }
    return {TransferSetKind::Full, TransferSetKind::Reduced};
}

inline DynamicTimeTable::Data loadDynamicTimeTable(const std::string& file,
                                                  const std::string_view label = "DynamicTimeTable",
                                                  const bool printInfo = true) {
    std::cout << "Loading " << label << "..." << std::endl;
    DynamicTimeTable::Data dynamicTimeTable(file);
    if (printInfo) {
        dynamicTimeTable.printInfo();
    }
    return dynamicTimeTable;
}

inline void printQueryDataSummary(const DynamicQueryData& queryData) {
    std::cout << "  Exported Routes: " << queryData.queryData.routeLabels.size() << std::endl;
    std::cout << "  Exported Trips: " << queryData.queryData.firstStopEventOfTrip.size() << std::endl;
    std::cout << "  Exported Events: " << queryData.queryData.eventLookup.size() << std::endl;
    std::cout << "  Exported Route Segments: " << queryData.queryData.routeSegments.size() << std::endl;
}

inline void printDegreeDistribution(const std::string_view label,
                                    const std::unordered_map<u_int64_t, u_int64_t>& distribution) {
    std::cout << "  " << label << ": \n";
    std::vector<std::pair<u_int64_t, u_int64_t>> sorted;
    sorted.reserve(distribution.size());
    u_int64_t sum = 0;
    for (const auto& entry : distribution) {
        sorted.emplace_back(entry);
        sum += entry.second;
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& entry : sorted) {
        const double percentage = sum == 0 ? 0.0 : (entry.second / double(sum)) * 100.0;
        std::cout << entry.first << ":" << entry.second << " " << percentage << "% " << std::endl;
    }
}

inline void printDegreeDistributions(const std::unordered_map<u_int64_t, u_int64_t>& out,
                                     const std::unordered_map<u_int64_t, u_int64_t>& in) {
    std::cout << "Degree distributions:\n";
    printDegreeDistribution("Outgoing", out);
    std::cout << "\n";
    printDegreeDistribution("Incoming", in);
}

template <typename Store>
inline std::size_t countTransferStoreEdges(Store& store) {
    auto& mutableStore = const_cast<std::remove_const_t<Store>&>(store);
    std::size_t edgeCount = 0;
    for (std::size_t i = 0; i < mutableStore.node_count(); ++i) {
        edgeCount += mutableStore.outgoing_sorted(PersistentStopEventId(i)).size();
    }
    return edgeCount;
}

template <typename Store>
inline void printTransferStoreSummary(Store& store, TransferUpdater& updater) {
    std::cout << "  Nodes: " << store.node_count() << std::endl;
    std::cout << "  Edges: " << countTransferStoreEdges(store) << std::endl;
    printDegreeDistributions(updater.store_.edgeDegreeDistrebutionOut(), updater.store_.edgeDegreeDistrebutionIn());
}

inline bool validateGenerators(const TripBased::Transfers& genA, const TripBased::Transfers& genB,
                               const DynamicTimeTable::Algo::DynamicQueryData& query_data) {
    auto diff = TripBased::compareTransfers(genA, genB);

    if (diff.areEqual()) {
        std::cout << "Topologies are identical!\n";
        return true;
    } else if (diff.isFirstSuperset()) {
        std::cout << "Generator A is a strict superset of Generator B.\n";
        std::cout << "Only in A: " << diff.onlyInFirst.size() << " edges.\n";
    } else if (diff.isSecondSuperset()) {
        std::cout << "Generator B is a strict superset of Generator A.\n";
        std::cout << "Only in B: " << diff.onlyInSecond.size() << " edges.\n";
    } else {
        std::cout << "Topologies diverge independently.\n";
        std::cout << "Only in A: " << diff.onlyInFirst.size() << "\n";
        std::cout << "Only in B: " << diff.onlyInSecond.size() << "\n";
    }

    if (!diff.onlyInFirst.empty()) {
        std::cout << "\nSample edges only in A:\n";
        for (size_t i = 0; i < std::min(diff.onlyInFirst.size(), size_t(100)); ++i) {
            const auto& edge = diff.onlyInFirst[i];
            auto p_from_event = query_data.flatToPersistentEvent[edge.from];
            auto p_to_event = query_data.flatToPersistentEvent[edge.to];
            auto from_trip = query_data.queryData.tripOfStopEvent[edge.from];
            auto to_trip = query_data.queryData.tripOfStopEvent[edge.to];
            auto p_from_trip = query_data.flatToPersistentTrip[from_trip];
            auto p_to_trip = query_data.flatToPersistentTrip[to_trip];
            auto from_route = query_data.queryData.routeOfTrip[from_trip];
            auto to_route = query_data.queryData.routeOfTrip[to_trip];
            auto p_from_route = query_data.flatToPersistentRoute[from_route];
            auto p_to_route = query_data.flatToPersistentRoute[to_route];
            auto from_stop_idx = edge.from - query_data.queryData.firstStopEventOfTrip[from_trip];
            auto to_stop_idx = edge.to - query_data.queryData.firstStopEventOfTrip[to_trip];
            /*
            auto from_stop = query_data.queryData.routeStopSequences[query_data.queryData.firstStopIdOfRoute[from_route] + from_stop_idx];
            auto to_stop = query_data.queryData.routeStopSequences[query_data.queryData.firstStopIdOfRoute[to_route] + to_stop_idx];
            auto from_arrival = query_data.queryData.eventArrTimes[edge.from];
            auto to_departure = query_data.queryData.eventDepTimes[edge.to];
*/

            // - Route:p`persistentRouteID`:f`flatRouteID` Trip:p``:f`` StopIndex: PersistentEvent ->
            // Route:p`persistentRouteID`:f`flatRouteID` Trip:p``:f`` StopIndex: PersistentEvent
            std::cout << "- Route:p" << p_from_route << ":f" << from_route << " Trip:p" << p_from_trip << ":f"
                      << from_trip << " StopIndex: " << from_stop_idx << " (Event:p" << p_from_event << ":f"
                      << edge.from << ") -> "
                      << "Route:p" << p_to_route << ":f" << to_route << " Trip:p" << p_to_trip << ":f" << to_trip
                      << " StopIndex: " << to_stop_idx << " (Event:p" << p_to_event << ":f" << edge.to << ")\n";

            /* std::cout << "- Route:p" << p_from_route << ":f" << from_route << " Trip:p" << p_from_trip << ":f"
                      << from_trip << " StopIndex: " << from_stop_idx << " (Event:p" << p_from_event << ":f"
                      << edge.from << ") with arrival at second: " << from_arrival << " at StopId " << from_stop << " -> "
                      << "Route:p" << p_to_route << ":f" << to_route << " Trip:p" << p_to_trip << ":f" << to_trip
                      << " StopIndex: " << to_stop_idx << " (Event:p" << p_to_event << ":f" << edge.to << ")" << " with departure at second: "
                      << to_departure << " at StopId " << to_stop << "\n";
                      */
        }
    }
    if (!diff.onlyInSecond.empty()) {
        std::cout << "\nSample edges only in B:\n";
        for (size_t i = 0; i < std::min(diff.onlyInSecond.size(), size_t(100)); ++i) {
            const auto& edge = diff.onlyInSecond[i];
            auto p_from_event = query_data.flatToPersistentEvent[edge.from];
            auto p_to_event = query_data.flatToPersistentEvent[edge.to];
            auto from_trip = query_data.queryData.tripOfStopEvent[edge.from];
            auto to_trip = query_data.queryData.tripOfStopEvent[edge.to];
            auto p_from_trip = query_data.flatToPersistentTrip[from_trip];
            auto p_to_trip = query_data.flatToPersistentTrip[to_trip];
            auto from_route = query_data.queryData.routeOfTrip[from_trip];
            auto to_route = query_data.queryData.routeOfTrip[to_trip];
            auto p_from_route = query_data.flatToPersistentRoute[from_route];
            auto p_to_route = query_data.flatToPersistentRoute[to_route];
            auto from_stop_idx = edge.from - query_data.queryData.firstStopEventOfTrip[from_trip];
            auto to_stop_idx = edge.to - query_data.queryData.firstStopEventOfTrip[to_trip];
            std::cout << "- Route:p" << p_from_route << ":f" << from_route << " Trip:p" << p_from_trip << ":f"
                      << from_trip << " StopIndex: " << from_stop_idx << " (Event:p" << p_from_event << ":f"
                      << edge.from << ") -> "
                      << "Route:p" << p_to_route << ":f" << to_route << " Trip:p" << p_to_trip << ":f" << to_trip
                      << " StopIndex: " << to_stop_idx << " (Event:p" << p_to_event << ":f" << edge.to << ")\n";
        }
    }
    return false;
}


inline bool validateDynamicQueryData(const DynamicQueryData& queryData,
                                     const DynamicTimeTable::Data& dynamicTimeTable,
                                     const std::string_view context) {
    const auto validation = queryData.validate(dynamicTimeTable);
    const std::string_view prefix = context.empty() ? std::string_view{} : context;
    const std::string_view separator = context.empty() ? std::string_view{} : std::string_view{" "};
    if (!validation.first) {
        std::cout << prefix << separator << "DynamicQueryData validation failed: " << validation.second << std::endl;
        return false;
    }
    std::cout << prefix << separator << "DynamicQueryData validation OK." << std::endl;
    return true;
}

inline void buildInitialTransfers(TransferUpdater& updater,
                                  const DynamicQueryData& queryData,
                                  const TransferSetKind kind,
                                  const int numberOfThreads) {
    updater.buildInitialFullTransfers(queryData);
    if (kind == TransferSetKind::Reduced) {
        updater.buildInitialMinimizedTransfers(queryData, numberOfThreads);
    }
}

inline void buildInitialTransfersForSelection(TransferUpdater& updater,
                                              const DynamicQueryData& queryData,
                                              const TransferSetSelection selection,
                                              const int numberOfThreads) {
    updater.buildInitialFullTransfers(queryData);
    if (selection != TransferSetSelection::Full) {
        updater.buildInitialMinimizedTransfers(queryData, numberOfThreads);
    }
}

inline TripBased::Transfers exportTransfers(const TransferUpdater& updater,
                                            const DynamicQueryData& queryData,
                                            const TransferSetKind kind,
                                            const int numberOfThreads) {
    if (kind == TransferSetKind::Full) {
        return updater.exportFullTransfers(queryData, numberOfThreads);
    }
    return updater.exportReducedTransfers(queryData, numberOfThreads);
}

inline std::size_t countModifiedStops(const DynamicTimeTable::PendingUpdates& updates) {
    std::size_t totalStopMods = 0;
    for (const auto& entry : updates.modifications) {
        totalStopMods += entry.second.size();
    }
    return totalStopMods;
}

inline void printSimulationSummary(const DynamicTimeTable::Algo::UpdateSimulationStats& simStats,
                                   const DynamicTimeTable::PendingUpdates& updates,
                                   const DynamicTimeTable::UpdateStatistics& updateStats,
                                   const DynamicTimeTable::ChangeSummary& changes) {
    std::cout << "\nSimulation summary" << std::endl;
    std::cout << "  Cancelled trips (sim): " << simStats.cancelledTrips << std::endl;
    std::cout << "  Delayed trips (sim): " << simStats.delayedTrips << std::endl;
    std::cout << "  Skipped trips (sim): " << simStats.skippedTrips << std::endl;
    std::cout << "  Modified stops (sim): " << simStats.modifiedStops << std::endl;

    std::cout << "\nPendingUpdates summary" << std::endl;
    std::cout << "  Cancellations: " << updates.cancellations.size() << std::endl;
    std::cout << "  Modified trips: " << updates.modifications.size() << std::endl;
    std::cout << "  Modified stops: " << countModifiedStops(updates) << std::endl;
    std::cout << "  Additions: " << updates.additions.size() << std::endl;

    std::cout << "\nUpdateStatistics" << std::endl;
    std::cout << "  Total updates: " << updateStats.totalUpdates << std::endl;
    std::cout << "  Successful: " << updateStats.successfulUpdates << std::endl;
    std::cout << "  Failed: " << updateStats.failedUpdates << std::endl;
    std::cout << "  Cancellations: " << updateStats.cancellations << std::endl;
    std::cout << "  Modifications: " << updateStats.modifications << std::endl;
    std::cout << "  Additions: " << updateStats.additions << std::endl;

    std::cout << "\nChangeSummary" << std::endl;
    std::cout << "  Cancelled trips: " << changes.cancelledTrips.size() << std::endl;
    std::cout << "  Added trips: " << changes.addedTrips.size() << std::endl;
    std::cout << "  Modified events: " << changes.modifiedEvents.size() << std::endl;
    std::cout << "  Trips with delayed arrivals: " << changes.tripsWithDelayedArrivals.size() << std::endl;
}

inline void printChangeDetails(const DynamicTimeTable::ChangeSummary& changes, std::ostream& out = std::cout) {
    out << "\nIn Detail:" << std::endl;
    out << "  Cancelled trips: " << std::endl;
    for (const auto& cancelledTrip : changes.cancelledTrips) {
        out << "    " << cancelledTrip.tripId << std::endl;
    }
    out << "  Trips to Rediscover due to Cancellation: " << std::endl;
    for (const auto& nextOfCancelledTrip : changes.tripsToRediscoverIncomingDueToCancellation) {
        out << "    " << nextOfCancelledTrip << std::endl;
    }
    out << "  Added trips: " << std::endl;
    for (const auto& addedTrip : changes.addedTrips) {
        out << "    " << addedTrip << std::endl;
    }
    out << "  Modified events: " << std::endl;
    for (const auto& modifiedEvent : changes.modifiedEvents) {
        out << "    " << modifiedEvent.first << std::endl;
    }
}

inline DynamicQueryData buildQueryData(const DynamicTimeTable::Data& dynamicTimeTable,
                                      const std::string_view label = "DynamicQueryData",
                                      const bool printSummary = false) {
    std::cout << "Building " << label << "..." << std::endl;
    auto timed = runTimed([&]() { return DynamicQueryData::buildFromDynamic(dynamicTimeTable); });
    std::cout << "Successfully built " << label << " in " << timed.duration << std::endl;
    if (printSummary) {
        printQueryDataSummary(timed.value);
    }
    return std::move(timed.value);
}

class QueryDataBuilder {
public:
    DynamicQueryData operator()(const DynamicTimeTable::Data& dynamicTimeTable,
                                const std::string_view label = "DynamicQueryData",
                                const bool printSummary = false) const {
        return buildQueryData(dynamicTimeTable, label, printSummary);
    }
};

class InitialTransferStoreBuilder {
public:
    InitialTransferStoreBuilder(const TransferSetSelection selection,
                                const int numberOfThreads,
                                const bool printSummary = false)
        : selection(selection), numberOfThreads(numberOfThreads), printSummary(printSummary) {}

    void operator()(TransferUpdater& updater, const DynamicQueryData& queryData) const {
        std::cout << "Building initial transfer store..." << std::endl;
        const auto duration = timeAction([&]() {
            buildInitialTransfersForSelection(updater, queryData, selection, numberOfThreads);
        });
        std::cout << "Initial transfer store built in " << duration << std::endl;
        if (printSummary) {
            printTransferStoreSummary(updater.store_, updater);
        }
    }

private:
    TransferSetSelection selection;
    int numberOfThreads;
    bool printSummary;
};

class InitialExportValidator {
public:
    InitialExportValidator(std::vector<TransferSetKind> transferSets, const int numberOfThreads)
        : transferSets(std::move(transferSets)), numberOfThreads(numberOfThreads) {}

    bool operator()(const TransferUpdater& updater, const DynamicQueryData& queryData) const {
        bool allValid = true;
        for (const TransferSetKind kind : transferSets) {
            std::cout << "Validating initial " << nameOf(kind) << " transfer export..." << std::endl;
            TripBased::Transfers initialTransfers = exportTransfers(updater, queryData, kind, numberOfThreads);
            std::cout << "  Initial " << nameOf(kind) << " edges: " << initialTransfers.labels.size() << std::endl;
            allValid = TripBased::validateTransfers(initialTransfers, queryData.queryData) && allValid;
        }
        return allValid;
    }

private:
    std::vector<TransferSetKind> transferSets;
    int numberOfThreads;
};

class UpdateGenerator {
public:
    explicit UpdateGenerator(DynamicTimeTable::Algo::UpdateSimulationConfig config) : config(std::move(config)) {}

    DynamicTimeTable::PendingUpdates operator()(DynamicTimeTable::Data& dynamicTimeTable,
                                                const int nowSeconds,
                                                DynamicTimeTable::Algo::UpdateSimulationStats* stats = nullptr) const {
        DynamicTimeTable::Algo::UpdateSimulator simulator(config);
        return simulator.generate(dynamicTimeTable, Time(nowSeconds), stats);
    }

private:
    DynamicTimeTable::Algo::UpdateSimulationConfig config;
};

inline TimedAppliedUpdate applyIncrementalUpdateTimed(DynamicTimeTable::Data& dynamicTimeTable,
                                                       TransferUpdater& transferUpdater,
                                                       const DynamicTimeTable::PendingUpdates& updates,
                                                       const int numberOfThreads) {
    auto timed = runTimed([&]() {
        const DynamicTimeTable::UpdateStatistics statistics =
            DynamicTimeTable::Algo::UpdatePipeline::applyUpdates(dynamicTimeTable, updates);
        auto queryData = DynamicQueryData::buildFromDynamic(dynamicTimeTable);
        const DynamicTimeTable::ChangeSummary& changes = dynamicTimeTable.getLatestChanges();
        transferUpdater.applyFullUpdates(changes, queryData, numberOfThreads);
        return AppliedUpdate(std::move(queryData), changes, statistics);
    });
    return {std::move(timed.value), timed.duration};
}

class IncrementalUpdateApplier {
public:
    explicit IncrementalUpdateApplier(const int numberOfThreads) : numberOfThreads(numberOfThreads) {}

    AppliedUpdate operator()(DynamicTimeTable::Data& dynamicTimeTable,
                             TransferUpdater& transferUpdater,
                             const DynamicTimeTable::PendingUpdates& updates) const {
        return applyIncrementalUpdateTimed(dynamicTimeTable, transferUpdater, updates, numberOfThreads).applied;
    }

private:
    int numberOfThreads;
};

class RebuildComparator {
public:
    RebuildComparator(std::vector<TransferSetKind> transferSets, const int numberOfThreads)
        : transferSets(std::move(transferSets)), numberOfThreads(numberOfThreads) {}

    bool operator()(TransferUpdater& incrementalUpdater,
                    const DynamicQueryData& updatedQueryData,
                    std::ostream* out = nullptr) const {
        bool allOk = true;
        for (const TransferSetKind kind : transferSets) {
            const ComparisonResult result = compareOneSet(incrementalUpdater, updatedQueryData, kind, out);
            allOk = allOk && result.transferValidation && result.matchesRebuild;
        }
        return allOk;
    }

private:
    ComparisonResult compareOneSet(TransferUpdater& incrementalUpdater,
                                   const DynamicQueryData& updatedQueryData,
                                   const TransferSetKind kind,
                                   std::ostream* out) const {
        std::ostream& log = out != nullptr ? *out : std::cout;
        log << "Validating " << nameOf(kind) << " incremental transfers..." << std::endl;

        TripBased::Transfers incrementalTransfers =
            exportTransfers(incrementalUpdater, updatedQueryData, kind, numberOfThreads);
        const bool valid = TripBased::validateTransfers(incrementalTransfers, updatedQueryData.queryData);

        TransferStoreType rebuildStore;
        TransferUpdater rebuildUpdater(rebuildStore);
        buildInitialTransfers(rebuildUpdater, updatedQueryData, kind, numberOfThreads);
        TripBased::Transfers rebuiltTransfers =
            exportTransfers(rebuildUpdater, updatedQueryData, kind, numberOfThreads);

        log << "  Incremental " << nameOf(kind) << " edges: " << incrementalTransfers.labels.size() << std::endl;
        log << "  Rebuilt " << nameOf(kind) << " edges: " << rebuiltTransfers.labels.size() << std::endl;
        log << "  Transfer validation: " << (valid ? "PASSED" : "FAILED") << std::endl;

        std::streambuf* coutBuf = nullptr;
        if (out != nullptr) {
            coutBuf = std::cout.rdbuf(out->rdbuf());
        }
        const bool matchesRebuild = validateGenerators(incrementalTransfers, rebuiltTransfers, updatedQueryData);
        if (coutBuf != nullptr) {
            std::cout.rdbuf(coutBuf);
        }

        return {valid, matchesRebuild, incrementalTransfers.labels.size(), rebuiltTransfers.labels.size()};
    }

    std::vector<TransferSetKind> transferSets;
    int numberOfThreads;
};

inline UpdateComparisonResult simulateApplyAndCompare(
    DynamicTimeTable::Data& dynamicTimeTable,
    TransferUpdater& transferUpdater,
    const DynamicTimeTable::Algo::UpdateSimulationConfig& simulationConfig,
    const int nowSeconds,
    const int numberOfThreads,
    const std::vector<TransferSetKind>& transferSets,
    std::ostream* out = nullptr) {
    DynamicTimeTable::Algo::UpdateSimulationStats simStats{};
    DynamicTimeTable::PendingUpdates updates = UpdateGenerator(simulationConfig)(dynamicTimeTable, nowSeconds, &simStats);
    AppliedUpdate applied = IncrementalUpdateApplier(numberOfThreads)(dynamicTimeTable, transferUpdater, updates);
    const bool ok = RebuildComparator(transferSets, numberOfThreads)(transferUpdater, applied.queryData, out);
    return {std::move(applied), std::move(updates), simStats, ok};
}

}  // namespace DynamicTransferScenarios
