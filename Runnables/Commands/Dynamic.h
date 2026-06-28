#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>

#include "../../Algorithms/DynamicTimeTable/BuildQueryData.h"
#include "../../Algorithms/DynamicTimeTable/Update.h"
#include "../../Algorithms/DynamicTimeTable/UpdateSimulation.h"
#include "../../DataStructures/DynamicTimeTable/Data.h"
#include "../../DataStructures/Graph/Graph.h"
#include "../../DataStructures/RAPTOR/Data.h"
#include "../../DataStructures/TransferStore/TransferStore.h"
#include "../../DataStructures/TripBased/Data.h"
#include "../../Shell/Shell.h"
#include "Algorithms/DynamicTB/preprocessing/TransferUpdate.h"
#include "Algorithms/TripBased/Preprocessing/StopEventGraphBuilder.h"

using namespace Shell;

// === Helpers ===

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
            auto from_stop = query_data.queryData.routeStopSequences[query_data.queryData.firstStopIdOfRoute[from_route] + from_stop_idx];
            auto to_stop = query_data.queryData.routeStopSequences[query_data.queryData.firstStopIdOfRoute[to_route] + to_stop_idx];
            auto from_arrival = query_data.queryData.eventArrTimes[edge.from];
            auto to_departure = query_data.queryData.eventDepTimes[edge.to];


            // - Route:p`persistentRouteID`:f`flatRouteID` Trip:p``:f`` StopIndex: PersistentEvent ->
            // Route:p`persistentRouteID`:f`flatRouteID` Trip:p``:f`` StopIndex: PersistentEvent
            std::cout << "- Route:p" << p_from_route << ":f" << from_route << " Trip:p" << p_from_trip << ":f"
                      << from_trip << " StopIndex: " << from_stop_idx << " (Event:p" << p_from_event << ":f"
                      << edge.from << ") with arrival at second: " << from_arrival << " at StopId " << from_stop << " -> "
                      << "Route:p" << p_to_route << ":f" << to_route << " Trip:p" << p_to_trip << ":f" << to_trip
                      << " StopIndex: " << to_stop_idx << " (Event:p" << p_to_event << ":f" << edge.to << ")" << " with departure at second: "
                      << to_departure << " at StopId " << to_stop << "\n";
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

inline void printDegreeDistributions(const std::unordered_map<u_int64_t, u_int64_t>& out, const std::unordered_map<u_int64_t, u_int64_t>& in) {
    std::cout << "Degree distributions:\n";
    std::cout << "  Outgoing: \n";
    // extract and sort by key
    std::vector<std::pair<u_int64_t, u_int64_t>> out_sorted;
    out_sorted.reserve(out.size());
    u_int64_t sum = 0;
    for (const auto& entry : out) {
        out_sorted.emplace_back(entry);
        sum += entry.second;
    }
    std::sort(out_sorted.begin(), out_sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& entry : out_sorted) {
        std::cout << entry.first << ":" << entry.second << " " << (entry.second / double(sum)) * 100 << "% " << std::endl;
    }

    std::cout << "\n  Incoming: \n";
    sum = 0;
    std::vector<std::pair<u_int64_t, u_int64_t>> in_sorted;
    in_sorted.reserve(in.size());
    for (const auto& entry : in) {
        in_sorted.emplace_back(entry);
        sum += entry.second;
    }
    std::sort(in_sorted.begin(), in_sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& entry : in_sorted) {
        std::cout << entry.first << ":" << entry.second << " " << (entry.second / double(sum)) * 100 << "% " << std::endl;
    }
}

class RAPTORToDynamic : public ParameterizedCommand {
public:
    RAPTORToDynamic(BasicShell& shell)
        : ParameterizedCommand(shell, "raptorToDynamic", "Converts a RAPTOR object into a DynamicTimeTable object.") {
        addParameter("Input binary (RAPTOR Data)");
        addParameter("Output binary (DynamicTimeTable data)");
    }

    virtual void execute() noexcept override {
        const std::string inputFile = getParameter("Input binary (RAPTOR Data)");
        const std::string outputFile = getParameter("Output binary (DynamicTimeTable data)");

        RAPTOR::Data data(inputFile);
        data.printInfo();

        DynamicTimeTable::Data dynamicTimeTable(data);
        dynamicTimeTable.printInfo();
        dynamicTimeTable.serialize(outputFile);
    }
};

class LoadAndApplyDynamicPartition : public ParameterizedCommand {
public:
    LoadAndApplyDynamicPartition(BasicShell& shell)
        : ParameterizedCommand(
              shell, "loadAndApplyDynamicPartition",
              "Loads a partition file generated by a black-box partitioner and applies the cell ids to the stops.") {
        addParameter("Input partition file");
        addParameter("Input binary (DynamicTimeTable Data)");
    }

    virtual void execute() noexcept override {
        const std::string partitionFile = getParameter("Input partition file");
        const std::string dynamicFile = getParameter("Input binary (DynamicTimeTable Data)");

        DynamicTimeTable::Data dynamicTimeTable(dynamicFile);
        dynamicTimeTable.printInfo();

        dynamicTimeTable.createCompactLayoutGraph();
        dynamicTimeTable.readPartitionFile(partitionFile);

        dynamicTimeTable.serialize(dynamicFile);
    }
};

class BuildDynamicQueryData : public ParameterizedCommand {
public:
    BuildDynamicQueryData(BasicShell& shell)
        : ParameterizedCommand(
              shell, "buildDynamicQueryData",
              "Loads a DynamicTimeTable and builds the QueryData structure to test export compilation.") {
        addParameter("Input binary (DynamicTimeTable Data)");
    }

    virtual void execute() noexcept override {
        const std::string dynamicFile = getParameter("Input binary (DynamicTimeTable Data)");

        std::cout << "Loading DynamicTimeTable..." << std::endl;
        DynamicTimeTable::Data dynamicTimeTable(dynamicFile);
        dynamicTimeTable.printInfo();

        std::cout << "Building DynamicQueryData..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto queryData = DynamicTimeTable::Algo::DynamicQueryData::buildFromDynamic(dynamicTimeTable);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Successfully built DynamicQueryData in " << duration << std::endl;

        std::cout << "  Exported Routes: " << queryData.queryData.routeLabels.size() << std::endl;
        std::cout << "  Exported Trips: " << queryData.queryData.firstStopEventOfTrip.size() << std::endl;
        std::cout << "  Exported Events: " << queryData.queryData.eventLookup.size() << std::endl;
        std::cout << "  Exported Route Segments: " << queryData.queryData.routeSegments.size() << std::endl;
        std::cout << "Validating DynamicQueryData..." << std::endl;
        const auto validation = queryData.validate(dynamicTimeTable);
        if (!validation.first) {
            std::cout << "DynamicQueryData validation failed: " << validation.second << std::endl;
        } else {
            std::cout << "DynamicQueryData validation OK." << std::endl;
        }
    }
};

class BuildInitialTransferStore : public ParameterizedCommand {
public:
    BuildInitialTransferStore(BasicShell& shell)
        : ParameterizedCommand(shell, "buildInitialTransferStore",
                               "Loads a DynamicTimeTable and builds the initial transfer store (full set).") {
        addParameter("Input binary (DynamicTimeTable Data)");
    }

    virtual void execute() noexcept override {
        const std::string dynamicFile = getParameter("Input binary (DynamicTimeTable Data)");

        std::cout << "Loading DynamicTimeTable..." << std::endl;
        DynamicTimeTable::Data dynamicTimeTable(dynamicFile);
        dynamicTimeTable.printInfo();

        std::cout << "Building DynamicQueryData..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto queryData = DynamicTimeTable::Algo::DynamicQueryData::buildFromDynamic(dynamicTimeTable);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "DynamicQueryData built in " << duration << std::endl;

        std::cout << "Building initial transfer store..." << std::endl;
        using TransferMeta = DynamicTB::Preprocessing::TransferMeta;
        using TransferStoreType = TransferStore<PersistentStopEventId, TransferMeta>;

        TransferStoreType store;
        DynamicTB::Preprocessing::TransferUpdate updater(store);

        start = std::chrono::high_resolution_clock::now();
        updater.buildInitialFullTransfers(queryData);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "Initial transfer store built in " << duration << std::endl;

        const std::size_t nodeCount = store.node_count();
        std::size_t edgeCount = 0;
        for (std::size_t i = 0; i < nodeCount; ++i) {
            edgeCount += store.outgoing_sorted(PersistentStopEventId(i)).size();
        }

        std::cout << "  Nodes: " << nodeCount << std::endl;
        std::cout << "  Edges: " << edgeCount << std::endl;

        printDegreeDistributions(updater.store_.edgeDegreeDistrebutionOut(), updater.store_.edgeDegreeDistrebutionIn());
    }
};

class SimulateDynamicUpdate : public ParameterizedCommand {
public:
    SimulateDynamicUpdate(BasicShell& shell)
        : ParameterizedCommand(
              shell, "simulateDynamicUpdates",
              "Simulates and applies dynamic updates, measuring execution time and printing a summary.") {
        addParameter("Input binary (DynamicTimeTable Data)");
        addParameter("Current time (seconds)");
        addParameter("Random seed", "42");
        addParameter("Expected cancellations", "0");
        addParameter("Expected delays", "0");
        addParameter("Expected skipped trips", "0");
        addParameter("Cancellation horizon (seconds)", "7200");
        addParameter("Skip horizon (seconds)", "7200");
        addParameter("Min delay (seconds)", "60");
        addParameter("Max delay (seconds)", "600");
        addParameter("Max skipped stops per trip", "1");
    }

    virtual void execute() noexcept override {
        const std::string dynamicFile = getParameter("Input binary (DynamicTimeTable Data)");
        const int nowSeconds = getParameter<int>("Current time (seconds)");

        DynamicTimeTable::Algo::UpdateSimulationConfig cfg;
        cfg.seed = static_cast<uint32_t>(getParameter<int>("Random seed"));
        cfg.cancellations.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected cancellations"));
        cfg.delays.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected delays"));
        cfg.skips.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected skipped trips"));
        cfg.cancellations.horizon = Time(getParameter<int>("Cancellation horizon (seconds)"));
        cfg.skips.horizon = Time(getParameter<int>("Skip horizon (seconds)"));
        cfg.delays.minInitialDelay = Time(getParameter<int>("Min delay (seconds)"));
        cfg.delays.maxInitialDelay = Time(getParameter<int>("Max delay (seconds)"));
        cfg.skips.maxSkippedStopsPerTrip = getParameter<int>("Max skipped stops per trip");

        std::cout << "Loading DynamicTimeTable..." << std::endl;
        DynamicTimeTable::Data dynamicTimeTable(dynamicFile);
        dynamicTimeTable.printInfo();
        auto queryData = DynamicTimeTable::Algo::DynamicQueryData::buildFromDynamic(dynamicTimeTable);

        std::cout << "Validating DynamicQueryData..." << std::endl;
        const auto validation = queryData.validate(dynamicTimeTable);
        if (!validation.first) {
            std::cout << "DynamicQueryData validation failed: " << validation.second << std::endl;
        } else {
            std::cout << "DynamicQueryData validation OK." << std::endl;
        }

        std::cout << "Building initial transfer store..." << std::endl;
        using TransferMeta = DynamicTB::Preprocessing::TransferMeta;
        using TransferStoreType = TransferStore<PersistentStopEventId, TransferMeta, 1024, false>;

        TransferStoreType store;
        DynamicTB::Preprocessing::TransferUpdate transferUpdater(store);

        auto start = std::chrono::high_resolution_clock::now();
        transferUpdater.buildInitialFullTransfers(queryData);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "Initial transfer store built in " << duration << std::endl;

        DynamicTimeTable::Algo::UpdateSimulator simulator(cfg);
        DynamicTimeTable::Algo::UpdateSimulationStats simStats{};
        DynamicTimeTable::PendingUpdates updates = simulator.generate(dynamicTimeTable, Time(nowSeconds), &simStats);

        std::size_t totalStopMods = 0;
        for (const auto& entry : updates.modifications) {
            totalStopMods += entry.second.size();
        }

        std::cout << "Applying updates..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        DynamicTimeTable::UpdateStatistics updateStats =
            DynamicTimeTable::Algo::UpdatePipeline::applyUpdates(dynamicTimeTable, updates);
        auto queryDataUpdated = DynamicTimeTable::Algo::DynamicQueryData::buildFromDynamic(dynamicTimeTable);
        const DynamicTimeTable::ChangeSummary& changes = dynamicTimeTable.getLatestChanges();
        transferUpdater.applyFullUpdates(changes, queryDataUpdated);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "Validating DynamicQueryData..." << std::endl;
        const auto validation2 = queryDataUpdated.validate(dynamicTimeTable);
        if (!validation2.first) {
            std::cout << "DynamicQueryData validation failed: " << validation2.second << std::endl;
        } else {
            std::cout << "DynamicQueryData validation OK." << std::endl;
        }

        std::cout << "Update pipeline finished in " << duration << std::endl;
        std::cout << "\nSimulation summary" << std::endl;
        std::cout << "  Cancelled trips (sim): " << simStats.cancelledTrips << std::endl;
        std::cout << "  Delayed trips (sim): " << simStats.delayedTrips << std::endl;
        std::cout << "  Skipped trips (sim): " << simStats.skippedTrips << std::endl;
        std::cout << "  Modified stops (sim): " << simStats.modifiedStops << std::endl;

        std::cout << "\nPendingUpdates summary" << std::endl;
        std::cout << "  Cancellations: " << updates.cancellations.size() << std::endl;
        std::cout << "  Modified trips: " << updates.modifications.size() << std::endl;
        std::cout << "  Modified stops: " << totalStopMods << std::endl;
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
};

class SimulateAndCompareTransferUpdate : public ParameterizedCommand {
public:
    SimulateAndCompareTransferUpdate(BasicShell& shell)
        : ParameterizedCommand(
              shell, "simulateAndCompareTransferUpdate",
              "Builds the dynamic full transfer store, compares it against the static non-route-pruned "
              "Trip-Based/TREX base generation path, simulates updates, applies incremental transfer updates, "
              "then compares incremental export against a full rebuild export.") {
        addParameter("Input binary (DynamicTimeTable Data)");
        addParameter("Input binary (RAPTOR Data)");
        addParameter("Current time (seconds)");
        addParameter("Random seed", "42");
        addParameter("Expected cancellations", "0");
        addParameter("Expected delays", "0");
        addParameter("Expected skipped trips", "0");
        addParameter("Cancellation horizon (seconds)", "7200");
        addParameter("Skip horizon (seconds)", "7200");
        addParameter("Min delay (seconds)", "60");
        addParameter("Max delay (seconds)", "600");
        addParameter("Max skipped stops per trip", "1");
        addParameter("Number of threads", "max");
        addParameter("Pin multiplier", "1");
    }

    virtual void execute() noexcept override {
        const std::string dynamicFile = getParameter("Input binary (DynamicTimeTable Data)");
        const std::string raptorFile = getParameter("Input binary (RAPTOR Data)");
        const int nowSeconds = getParameter<int>("Current time (seconds)");

        DynamicTimeTable::Algo::UpdateSimulationConfig cfg;
        cfg.seed = static_cast<uint32_t>(getParameter<int>("Random seed"));
        cfg.cancellations.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected cancellations"));
        cfg.delays.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected delays"));
        cfg.skips.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected skipped trips"));
        cfg.cancellations.horizon = Time(getParameter<int>("Cancellation horizon (seconds)"));
        cfg.skips.horizon = Time(getParameter<int>("Skip horizon (seconds)"));
        cfg.delays.minInitialDelay = Time(getParameter<int>("Min delay (seconds)"));
        cfg.delays.maxInitialDelay = Time(getParameter<int>("Max delay (seconds)"));
        cfg.skips.maxSkippedStopsPerTrip = getParameter<int>("Max skipped stops per trip");
        const int pinMultiplier = getParameter<int>("Pin multiplier");

        const int numberOfThreads = getNumberOfThreads();
        using TransferMeta = DynamicTB::Preprocessing::TransferMeta;
        using TransferStoreType = TransferStore<PersistentStopEventId, TransferMeta>;

        std::cout << "Loading DynamicTimeTable..." << std::endl;
        DynamicTimeTable::Data dynamicTimeTable(dynamicFile);
        dynamicTimeTable.printInfo();

        std::cout << "Building initial DynamicQueryData..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto queryData = DynamicTimeTable::Algo::DynamicQueryData::buildFromDynamic(dynamicTimeTable);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Initial DynamicQueryData built in " << duration << std::endl;

        const auto initialValidation = queryData.validate(dynamicTimeTable);
        if (!initialValidation.first) {
            std::cout << "Initial DynamicQueryData validation failed: " << initialValidation.second << std::endl;
            //    return;
        }

        std::cout << "Building initial dynamic full transfer store..." << std::endl;
        TransferStoreType store;
        DynamicTB::Preprocessing::TransferUpdate transferUpdater(store);

        start = std::chrono::high_resolution_clock::now();
        transferUpdater.buildInitialFullTransfers(queryData);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Initial dynamic full transfer store built in " << duration << std::endl;

        std::cout << "Exporting initial dynamic full transfers..." << std::endl;
        const TripBased::Transfers initialDynamicTransfers = transferUpdater.exportFullTransfers(queryData, numberOfThreads);
        std::cout << "  Dynamic exported edges: " << initialDynamicTransfers.labels.size() << std::endl;

        std::cout << "Validate base transerfs:" << std::endl;
        validateTransfers(initialDynamicTransfers, queryData.queryData);

        /* std::cout << "Building static non-minimized full transfers..." << std::endl;
        RAPTOR::Data raptorData(raptorFile);
        TripBased::Data staticTripData(raptorData);

        start = std::chrono::high_resolution_clock::now();
        TripBased::ComputeFullStopEventGraph(staticTripData, numberOfThreads, pinMultiplier);
        stop = std::chrono::high_resolution_clock::now();

        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Static non-minimized full transfer graph built in " << duration << std::endl;

        const auto staticInitialTransfers = TripBased::Transfers(staticTripData);
        std::cout << "  Static exported edges: " << staticInitialTransfers.labels.size() << std::endl;

        std::cout << "\nInitial comparison: dynamic full store export vs static non-minimized full base path"
                  << std::endl;
        validateGenerators(initialDynamicTransfers, staticInitialTransfers, queryData); */

        DynamicTimeTable::Algo::UpdateSimulator simulator(cfg);
        DynamicTimeTable::Algo::UpdateSimulationStats simStats{};
        DynamicTimeTable::PendingUpdates updates = simulator.generate(dynamicTimeTable, Time(nowSeconds), &simStats);

        std::size_t totalStopMods = 0;
        for (const auto& entry : updates.modifications) {
            totalStopMods += entry.second.size();
        }

        std::cout << "\nApplying simulated timetable updates..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        const DynamicTimeTable::UpdateStatistics updateStats =
            DynamicTimeTable::Algo::UpdatePipeline::applyUpdates(dynamicTimeTable, updates);
        auto updatedQueryData = DynamicTimeTable::Algo::DynamicQueryData::buildFromDynamic(dynamicTimeTable);
        const DynamicTimeTable::ChangeSummary& changes = dynamicTimeTable.getLatestChanges();
        transferUpdater.applyFullUpdates(changes, updatedQueryData);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Incremental timetable + transfer update finished in " << duration << std::endl;

        const auto updatedValidation = updatedQueryData.validate(dynamicTimeTable);
        if (!updatedValidation.first) {
            std::cout << "Updated DynamicQueryData validation failed: " << updatedValidation.second << std::endl;
            // return;
        }

        std::cout << "Exporting incrementally updated full transfers..." << std::endl;
        const TripBased::Transfers incrementalTransfers = transferUpdater.exportFullTransfers(updatedQueryData, numberOfThreads);
        std::cout << "  Incremental exported edges: " << incrementalTransfers.labels.size() << std::endl;

        std::cout << "Rebuilding full transfer store from updated timetable..." << std::endl;
        TransferStoreType rebuildStore;
        DynamicTB::Preprocessing::TransferUpdate rebuildUpdater(rebuildStore);

        start = std::chrono::high_resolution_clock::now();
        rebuildUpdater.buildInitialFullTransfers(updatedQueryData);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Updated full transfer rebuild finished in " << duration << std::endl;

        std::cout << "Exporting rebuilt full transfers..." << std::endl;
        const TripBased::Transfers rebuiltTransfers = rebuildUpdater.exportFullTransfers(updatedQueryData, numberOfThreads);
        std::cout << "  Rebuilt exported edges: " << rebuiltTransfers.labels.size() << std::endl;

        std::cout << "\nUpdated-state comparison: incremental export vs full rebuild export" << std::endl;
        validateGenerators(incrementalTransfers, rebuiltTransfers, updatedQueryData);

        std::cout << "\nSimulation summary" << std::endl;
        std::cout << "  Cancelled trips (sim): " << simStats.cancelledTrips << std::endl;
        std::cout << "  Delayed trips (sim): " << simStats.delayedTrips << std::endl;
        std::cout << "  Skipped trips (sim): " << simStats.skippedTrips << std::endl;
        std::cout << "  Modified stops (sim): " << simStats.modifiedStops << std::endl;

        std::cout << "\nPendingUpdates summary" << std::endl;
        std::cout << "  Cancellations: " << updates.cancellations.size() << std::endl;
        std::cout << "  Modified trips: " << updates.modifications.size() << std::endl;
        std::cout << "  Modified stops: " << totalStopMods << std::endl;
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

        std::cout << "\nIn Detail:" << std::endl;
        std::cout << "  Cancelled trips: " << std::endl;
        for (const auto& cancelledTrip : changes.cancelledTrips) {
            std::cout << "    " << cancelledTrip.tripId << std::endl;
        }
        std::cout << "  Trips to Rediscover due to Cancellation: " << std::endl;
        for (const auto& nextOfCancelledTrip : changes.tripsToRediscoverIncomingDueToCancellation) {
            std::cout << "    " << nextOfCancelledTrip << std::endl;
        }
        std::cout << "  Added trips: " << std::endl;
        for (const auto& addedTrip : changes.addedTrips) {
            std::cout << "    " << addedTrip << std::endl;
        }
        std::cout << "  Modified events: " << std::endl;
        for (const auto& modifiedEvent : changes.modifiedEvents) {
            std::cout << "    " << modifiedEvent.first << std::endl;
        }
    }

private:
    inline int getNumberOfThreads() const noexcept {
        if (getParameter("Number of threads") == "max") {
            return numberOfCores();
        } else {
            return getParameter<int>("Number of threads");
        }
    }
};

class SimulateAndCompareTransferUpdates : public ParameterizedCommand {
public:
    SimulateAndCompareTransferUpdates(BasicShell& shell)
        : ParameterizedCommand(
              shell, "simulateAndCompareTransferUpdates",
              "Builds base transfer set, then runs simulation updates in a loop with "
              "different seeds using serialization to reset the baseline.") {
        addParameter("Input binary (DynamicTimeTable Data)");
        addParameter("Input binary (RAPTOR Data)");
        addParameter("Current time (seconds)", "28800");
        addParameter("Base Random seed", "1");
        addParameter("Iterations", "200");
        addParameter("Output File Path", "simulation_output.txt");
        addParameter("Temp Store Path", "base_store.tmp");
        addParameter("Expected cancellations", "100");
        addParameter("Expected delays", "100");
        addParameter("Expected skipped trips", "100");
        addParameter("Cancellation horizon (seconds)", "7200");
        addParameter("Skip horizon (seconds)", "7200");
        addParameter("Min delay (seconds)", "60");
        addParameter("Max delay (seconds)", "600");
        addParameter("Max skipped stops per trip", "1");
        addParameter("Number of threads", "max");
        addParameter("Pin multiplier", "1");
    }

    virtual void execute() noexcept override {
        const std::string dynamicFile = getParameter("Input binary (DynamicTimeTable Data)");
        const int nowSeconds = getParameter<int>("Current time (seconds)");
        const uint32_t baseSeed = static_cast<uint32_t>(getParameter<int>("Base Random seed"));
        const int iterations = getParameter<int>("Iterations");
        const std::string outFilePath = getParameter("Output File Path");
        const std::string tempStorePath = getParameter("Temp Store Path");

        DynamicTimeTable::Algo::UpdateSimulationConfig cfg;
        cfg.cancellations.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected cancellations"));
        cfg.delays.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected delays"));
        cfg.skips.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected skipped trips"));
        cfg.cancellations.horizon = Time(getParameter<int>("Cancellation horizon (seconds)"));
        cfg.skips.horizon = Time(getParameter<int>("Skip horizon (seconds)"));
        cfg.delays.minInitialDelay = Time(getParameter<int>("Min delay (seconds)"));
        cfg.delays.maxInitialDelay = Time(getParameter<int>("Max delay (seconds)"));
        cfg.skips.maxSkippedStopsPerTrip = getParameter<int>("Max skipped stops per trip");

        auto pinMultiplier = getParameter<int>("Pin multiplier");
        using TransferMeta = DynamicTB::Preprocessing::TransferMeta;
        using TransferStoreType = TransferStore<PersistentStopEventId, TransferMeta>;

        std::cout << "Loading Base DynamicTimeTable..." << std::endl;
        DynamicTimeTable::Data baseDynamicTimeTable(dynamicFile);

        std::cout << "Building initial Base DynamicQueryData..." << std::endl;
        auto baseQueryData =
            DynamicTimeTable::Algo::DynamicQueryData::buildFromDynamic(baseDynamicTimeTable);

        const auto initialValidation = baseQueryData.validate(baseDynamicTimeTable);
        if (!initialValidation.first) {
            std::cout << "Initial DynamicQueryData validation failed: " << initialValidation.second << std::endl;
        }

        std::cout << "Building initial base dynamic full transfer store..." << std::endl;
        TransferStoreType baseStore;
        DynamicTB::Preprocessing::TransferUpdate baseTransferUpdater(baseStore);
        baseTransferUpdater.buildInitialFullTransfers(baseQueryData);

        std::cout << "Validate base transerfs:" << std::endl;
        TripBased::Transfers initialDynamicTransfers = baseTransferUpdater.exportFullTransfers(baseQueryData, getNumberOfThreads());
        validateTransfers(initialDynamicTransfers, baseQueryData.queryData);

        std::cout << "Serializing baseline store to disk..." << std::endl;
        baseStore.serialize(tempStorePath);

        std::ofstream outFile(outFilePath);
        if (!outFile) {
            std::cerr << "Failed to open output file: " << outFilePath << std::endl;
            return;
        }

        std::cout << "Starting simulation loop across " << iterations << " iterations..." << std::endl;
        bool divergenceFound = false;

        for (int i = 0; i < iterations; ++i) {
            uint32_t currentSeed = baseSeed + i;
            cfg.seed = currentSeed;

            std::cout << "\rRunning iteration " << (i + 1) << "/" << iterations << " (Seed: " << currentSeed << ")..." << std::flush;

            // Deep-copy the timetable via standard copy-constructor
            DynamicTimeTable::Data dynamicTimeTableCopy = baseDynamicTimeTable;

            // Reconstruct the base store completely fresh using your file constructor
            TransferStoreType storeCopy(tempStorePath);
            DynamicTB::Preprocessing::TransferUpdate transferUpdater(storeCopy);

            // Generate updates with the loop-specific seed
            DynamicTimeTable::Algo::UpdateSimulator simulator(cfg);
            DynamicTimeTable::Algo::UpdateSimulationStats simStats{};
            DynamicTimeTable::PendingUpdates updates = simulator.generate(dynamicTimeTableCopy, Time(nowSeconds), &simStats);

            // Apply incremental updates
            DynamicTimeTable::Algo::UpdatePipeline::applyUpdates(dynamicTimeTableCopy, updates);
            auto updatedQueryData = DynamicTimeTable::Algo::DynamicQueryData::buildFromDynamic(dynamicTimeTableCopy);
            const DynamicTimeTable::ChangeSummary& changes = dynamicTimeTableCopy.getLatestChanges();

            transferUpdater.applyFullUpdates(changes, updatedQueryData);
            TripBased::Transfers incrementalTransfers = transferUpdater.exportFullTransfers(updatedQueryData, getNumberOfThreads());

            // Rebuild from scratch to evaluate convergence
            TransferStoreType rebuildStore;
            DynamicTB::Preprocessing::TransferUpdate rebuildUpdater(rebuildStore);
            rebuildUpdater.buildInitialFullTransfers(updatedQueryData);
            const TripBased::Transfers rebuiltTransfers = rebuildUpdater.exportFullTransfers(updatedQueryData, getNumberOfThreads());

            outFile << "=== Iteration: " << (i + 1) << " | Seed: " << currentSeed << " ===\n";
            outFile << "Incremental edges: " << incrementalTransfers.labels.size()
                    << " | Rebuilt edges: " << rebuiltTransfers.labels.size() << "\n";

            std::streambuf* coutBuf = std::cout.rdbuf();
            std::cout.rdbuf(outFile.rdbuf());
            std::cout << "Validate incremental transerfs:" << std::endl;
            bool valid = TripBased::validateTransfers(incrementalTransfers, updatedQueryData.queryData);
            std::cout << "Incremental Transfers validation: " << (valid ? "PASSED" : "FAILED") << "\n";

            // Safety size check
            bool isConsistent = validateGenerators(incrementalTransfers, rebuiltTransfers, updatedQueryData);
            std::cout.rdbuf(coutBuf);

            if (!isConsistent) {
                divergenceFound = true;

                std::cout << "\n\n!!! DIVERGENCE DETECTED !!!" << std::endl;
                std::cout << "Replication Seed: " << currentSeed << std::endl;
                std::cout << "Incremental Edges: " << incrementalTransfers.labels.size() << std::endl;
                std::cout << "Rebuilt Edges: "     << rebuiltTransfers.labels.size() << std::endl;
                std::cout << "Check file details at: " << outFilePath << std::endl;

                outFile << "\n[!] CRITICAL DIVERGENCE FOUND FOR SEED: " << currentSeed << "\n";

                std::streambuf* coutBuf = std::cout.rdbuf();
                std::cout.rdbuf(outFile.rdbuf());

                std::cout << "\nChangeSummary" << std::endl;
                std::cout << "  Cancelled trips: " << changes.cancelledTrips.size() << std::endl;
                std::cout << "  Added trips: " << changes.addedTrips.size() << std::endl;
                std::cout << "  Modified events: " << changes.modifiedEvents.size() << std::endl;
                std::cout << "  Trips with delayed arrivals: " << changes.tripsWithDelayedArrivals.size() << std::endl;

                std::cout << "\nIn Detail:" << std::endl;
                std::cout << "  Cancelled trips: " << std::endl;
                for (const auto& cancelledTrip : changes.cancelledTrips) {
                    std::cout << "    " << cancelledTrip.tripId << std::endl;
                }
                std::cout << "  Trips to Rediscover due to Cancellation: " << std::endl;
                for (const auto& nextOfCancelledTrip : changes.tripsToRediscoverIncomingDueToCancellation) {
                    std::cout << "    " << nextOfCancelledTrip << std::endl;
                }
                std::cout << "  Added trips: " << std::endl;
                for (const auto& addedTrip : changes.addedTrips) {
                    std::cout << "    " << addedTrip << std::endl;
                }
                std::cout << "  Modified events: " << std::endl;
                for (const auto& modifiedEvent : changes.modifiedEvents) {
                    std::cout << "    " << modifiedEvent.first << std::endl;
                }

                std::cout.rdbuf(coutBuf);
            }

            outFile << "Status: Success (No Issues Found)\n\n";
        }

        // Cleanup temporary binary tracking files
        std::remove(tempStorePath.c_str());

        if (!divergenceFound) {
            std::cout << "\nAll iterations completed successfully. No transfer divergences found." << std::endl;
        }
    }


private:
    inline int getNumberOfThreads() const noexcept {
        if (getParameter("Number of threads") == "max") {
            return numberOfCores();
        } else {
            return getParameter<int>("Number of threads");
        }
    }
};