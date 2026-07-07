#pragma once

#include <algorithm>
#include <cstddef>
#include <string>

#include "../../Helpers/MultiThreading.h"
#include "../../Helpers/String/String.h"
#include "../../Shell/Shell.h"
#include "DynamicTransferScenarios.h"

class DynamicTransferCommandBase : public Shell::ParameterizedCommand {
protected:
    using TransferSetConfig = DynamicTransferScenarios::TransferSetConfig;
    using TransferSetSelection = DynamicTransferScenarios::TransferSetSelection;

    DynamicTransferCommandBase(Shell::BasicShell& shell,
                               const std::string& name,
                               const std::string& description)
        : Shell::ParameterizedCommand(shell, name, description) {}

    void addSimulationParameters(const std::string& defaultSeed = "42",
                                 const std::string& defaultCancellations = "0",
                                 const std::string& defaultDelays = "0",
                                 const std::string& defaultSkippedTrips = "0") {
        addParameter("Random seed", defaultSeed);
        addSimulationRateParameters(defaultCancellations, defaultDelays, defaultSkippedTrips);
    }

    void addSimulationRateParameters(const std::string& defaultCancellations,
                                     const std::string& defaultDelays,
                                     const std::string& defaultSkippedTrips) {
        addParameter("Expected cancellations", defaultCancellations);
        addParameter("Expected delays", defaultDelays);
        addParameter("Expected skipped trips", defaultSkippedTrips);
        addParameter("Cancellation horizon (seconds)", "7200");
        addParameter("Skip horizon (seconds)", "7200");
        addParameter("Min delay (seconds)", "60");
        addParameter("Max delay (seconds)", "600");
        addParameter("Max skipped stops per trip", "1");
    }

    void addTransferSetParameters() {
        addParameter("Transfer set", "both");
        addParameter("Number of threads", "max");
    }

    int numberOfThreads() const {
        const std::string value = getParameter("Number of threads");
        if (value == "max") return numberOfCores();
        return std::max(1, String::lexicalCast<int>(value));
    }

    TransferSetSelection transferSetSelection() const {
        const std::string value = getParameter("Transfer set");
        if (value == "full") return TransferSetSelection::Full;
        if (value == "reduced") return TransferSetSelection::Reduced;
        if (value == "both") return TransferSetSelection::Both;
        std::cout << "Unknown Transfer set '" << value << "', using 'both'. Expected full, reduced, or both." << std::endl;
        return TransferSetSelection::Both;
    }

    TransferSetConfig transferSetConfig() const {
        const TransferSetSelection selection = transferSetSelection();
        return {numberOfThreads(), selection, DynamicTransferScenarios::expandTransferSetSelection(selection)};
    }

    DynamicTimeTable::Algo::UpdateSimulationConfig simulationConfig() const {
        DynamicTimeTable::Algo::UpdateSimulationConfig cfg;
        cfg.seed = static_cast<uint32_t>(getParameter<int>("Random seed"));
        fillSimulationRates(cfg);
        return cfg;
    }

    DynamicTimeTable::Algo::UpdateSimulationConfig simulationConfigWithSeed(const uint32_t seed) const {
        DynamicTimeTable::Algo::UpdateSimulationConfig cfg;
        cfg.seed = seed;
        fillSimulationRates(cfg);
        return cfg;
    }

private:
    void fillSimulationRates(DynamicTimeTable::Algo::UpdateSimulationConfig& cfg) const {
        cfg.cancellations.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected cancellations"));
        cfg.delays.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected delays"));
        cfg.skips.expectedCount = static_cast<std::size_t>(getParameter<int>("Expected skipped trips"));
        cfg.cancellations.horizon = Time(getParameter<int>("Cancellation horizon (seconds)"));
        cfg.skips.horizon = Time(getParameter<int>("Skip horizon (seconds)"));
        cfg.delays.minInitialDelay = Time(getParameter<int>("Min delay (seconds)"));
        cfg.delays.maxInitialDelay = Time(getParameter<int>("Max delay (seconds)"));
        cfg.skips.maxSkippedStopsPerTrip = getParameter<int>("Max skipped stops per trip");
    }
};
