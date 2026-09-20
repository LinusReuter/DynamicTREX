#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <optional>
#include <utility>
#include <vector>

#include "../../Algorithms/DynamicTB/customization/CellBorderIndex.h"
#include "../../Algorithms/DynamicTB/customization/CellIndices.h"
#include "../../Algorithms/DynamicTB/customization/CellStopIndex.h"
#include "../../Algorithms/DynamicTB/customization/CustomizationDriver.h"
#include "../../Algorithms/DynamicTB/customization/RankComparison.h"
#include "../../DataStructures/TREX/TREXData.h"
#include "../../Helpers/String/String.h"
#include "../../Helpers/Timer.h"
#include "DynamicTransferCommandBase.h"

using namespace Shell;

namespace DynamicCustomizationCommands {

using namespace DynamicTB::Customization;
using namespace DynamicTransferScenarios;

/**
 * How many transfers ended up at each rank -- the one-line summary of a customization.
 *
 * Takes the ranks rather than a container so the CSR (`EdgeLabel::getRank`) and a flattened
 * `RankedEdge` list (`.rank`) share one printer and one bucket convention: 32 buckets, which is
 * every value a rank byte can carry that a query could act on.
 */
template <typename Range, typename RankOf>
inline void printRankHistogram(const std::string& label, const Range& range, const RankOf& rankOf,
                               std::ostream& out = std::cout) {
    std::vector<std::size_t> histogram(32, 0);
    std::size_t total = 0;
    for (const auto& element : range) {
        ++histogram[rankOf(element) & 31];
        ++total;
    }
    out << label << " over " << total << " transfers\n";
    for (std::size_t rank = 0; rank < histogram.size(); ++rank) {
        if (histogram[rank] == 0) continue;
        const double share = total == 0 ? 0.0 : (100.0 * histogram[rank]) / total;
        out << "  rank " << rank << ": " << histogram[rank] << " (" << share << "%)\n";
    }
}

inline void printCsrRankHistogram(const TripBased::Transfers& csr, std::ostream& out = std::cout) {
    printRankHistogram("Rank distribution", csr.labels, [](const auto& label) { return label.getRank(); }, out);
}

inline void printRankedEdgeHistogram(const std::string& label, const std::vector<RankedEdge>& edges,
                                     std::ostream& out = std::cout) {
    printRankHistogram(label, edges, [](const RankedEdge& edge) { return edge.rank; }, out);
}

inline void printRankMismatches(const RankComparison& comparison, std::ostream& out = std::cout) {
    out << "  Edges compared: " << comparison.edgesCompared << "\n";
    out << "  Mismatches: " << comparison.mismatches << " (under-approximated " << comparison.underApproximated
        << ", over-approximated " << comparison.overApproximated << ")\n";
    if (comparison.underApproximated > 0) {
        out << "  WARNING: an under-approximated rank prunes a transfer the query still needs.\n";
    }
    for (const RankMismatch& sample : comparison.samples) {
        out << "    event " << sample.from << " -> " << sample.to << ": " << static_cast<int>(sample.leftRank)
            << " vs " << static_cast<int>(sample.rightRank) << "\n";
    }
}

/**
 * Build the store and export the reduced, ranked CSR the customization consumes -- the same
 * artifact the per-minute pipeline already produces, which is the whole reason customization
 * adds no export pass.
 */
template <typename Updater>
inline TripBased::Transfers buildReducedExport(Updater& updater, const DynamicQueryData& queryData,
                                               const int threads) {
    std::cout << "Building initial transfer store..." << std::endl;
    Timer timer;
    updater.buildInitialFullTransfers(queryData, threads);
    updater.buildInitialMinimizedTransfers(queryData, threads);
    std::cout << "  built in " << timer.elapsedMilliseconds() << " ms" << std::endl;

    timer.restart();
    TripBased::Transfers csr = updater.exportReducedTransfers(queryData, threads);
    std::cout << "Exported " << csr.labels.size() << " reduced transfers in " << timer.elapsedMilliseconds() << " ms"
              << std::endl;
    return csr;
}

/**
 * What one incremental customization step did, and -- when the step was checked -- how its ranks
 * compare to a full customization of the same export.
 */
struct CustomizationStepResult {
    std::size_t rankChanges = 0;
    std::size_t seedsDirect = 0;
    std::size_t seedsStructural = 0;
    std::size_t reducedEdges = 0;
    bool checked = false;
    RankComparison comparison{};

    bool ok() const noexcept { return !checked || comparison.identical(); }
};

/**
 * @brief One store, one pair of cell indices, one customization driver -- the state a Dynamic
 * TREX run carries from minute to minute.
 *
 * The store is the only persistent rank state: the CSR is rebuilt from it every minute, so each
 * step exports, customizes the export in place, and writes the decisions back sparsely. That
 * round trip is what makes the *next* minute's ranks the previous minute's result, and it is the
 * reason the write-back has set semantics -- the incremental customization lowers ranks as well as
 * raising them.
 *
 * The updater is instantiated with `AffectedEventCollector`, which is what turns the level-0
 * affected set from a discarded by-product of minimization into the customization's direct seed.
 */
class CustomizationHarness {
public:
    /**
     * `keepReference` decides whether the reference customization driver survives setup. The
     * initial full customization always needs one; a measurement timeline does not, and a
     * `FullCustomizationDriver` holds
     * `threads x 4|trips| + |edges|` bytes of scratch that would otherwise sit in the memory
     * figures the measurement run reports.
     */
    CustomizationHarness(const int threads, const bool keepReference)
        : threads_(threads),
          updater_(store_),
          incremental_(threads),
          full_(std::in_place, threads),
          keepReference_(keepReference),
          csr_({}, {}, {}) {}

    TransferStoreType& store() noexcept { return store_; }
    CustomizingTransferUpdater& updater() noexcept { return updater_; }
    const IncrementalCustomizationDriver& incremental() const noexcept { return incremental_; }

    /**
     * Build the store and bring it to a *fully customized* state. A timeline that started from
     * uncustomized ranks would measure the incremental driver against a reference it never had a
     * chance to reproduce, so this is part of the setup, not part of the measurement.
     */
    void buildInitialState(const DynamicTimeTable::Data& data, const DynamicQueryData& queryData) {
        csr_ = buildReducedExport(updater_, queryData, threads_);

        Timer timer;
        indices_.build(data);
        indices_.sync(data);
        std::cout << "Cell indices built in " << timer.elapsedMilliseconds() << " ms ("
                  << String::bytesToString(indices_.byteSize()) << ")" << std::endl;

        timer.restart();
        const CellNetwork net = indices_.bind(data, queryData, csr_);
        full_->run(net);
        updater_.applyRankChanges(full_->changes(), threads_);
        std::cout << "Initial full customization: " << full_->changes().size() << " rank changes in "
                  << timer.elapsedMilliseconds() << " ms" << std::endl;
        printCsrRankHistogram(csr_);
        if (!keepReference_) full_.reset();
    }

    /**
     * Customize one applied update. Call after the timetable update and the transfer update have
     * run; `changes` is that update's `ChangeSummary` and the affected set is read off the
     * updater.
     *
     * With `check` on, a full customization of the same export is run afterwards and diffed
     * against the incremental result edge by edge, then the incremental ranks are restored so the
     * timeline continues from the state it actually produced -- a check must not repair what it
     * is checking.
     */
    CustomizationStepResult customizeStep(const DynamicTimeTable::Data& data, const DynamicQueryData& queryData,
                                          const DynamicTimeTable::ChangeSummary& changes, PhaseTimings& phases,
                                          const bool check) {
        // Routes minted by this update must be filed before any worklist is expanded: a route the
        // border index has never seen contributes no search seeds, i.e. under-approximated ranks.
        indices_.sync(data);
        csr_ = updater_.exportReducedTransfers(queryData, threads_, &phases);
        const CellNetwork net = indices_.bind(data, queryData, csr_);

        incremental_.seeds().bind(&updater_.affectedEvents(), &changes);
        Timer timer;
        incremental_.run(net);
        phases.customization += std::chrono::microseconds(static_cast<long long>(timer.elapsedMicroseconds()));

        timer.restart();
        updater_.applyRankChanges(incremental_.changes(), threads_);
        phases.rankWriteBack += std::chrono::microseconds(static_cast<long long>(timer.elapsedMicroseconds()));

        CustomizationStepResult result;
        result.rankChanges = incremental_.changes().size();
        result.seedsDirect = incremental_.seeds().directSeeds();
        result.seedsStructural = incremental_.seeds().structuralSeeds();
        result.reducedEdges = csr_.labels.size();
        if (!check) return result;

        // Scratch, not locals: at country scale each of these is a multi-MB allocation that would
        // otherwise be made and freed every checked minute.
        snapshotRanksInto(csr_, incrementalRanks_);
        // The reference customization's own change list is never read -- only the ranks it leaves
        // in the CSR are -- so discarding it saves a drain and a sort proportional to the export.
        full_->run(net, DynamicTB::Customization::ChangeOutput::Discard);
        snapshotRanksInto(csr_, referenceRanks_);
        restoreRanks(csr_, incrementalRanks_);
        // Left is the incremental result, right the full customization: `underApproximated` then
        // counts the edges the incremental driver ranked *below* the exact value -- the ones a
        // query would wrongly prune.
        result.comparison = compareRankSnapshot(csr_, referenceRanks_);
        result.checked = true;
        return result;
    }

private:
    int threads_;
    TransferStoreType store_;
    CustomizingTransferUpdater updater_;
    CellIndices indices_;
    IncrementalCustomizationDriver incremental_;
    std::optional<FullCustomizationDriver> full_;
    bool keepReference_;
    TripBased::Transfers csr_;
    std::vector<std::uint8_t> incrementalRanks_;
    std::vector<std::uint8_t> referenceRanks_;
};

inline void printStepResult(const CustomizationStepResult& result, std::ostream& out) {
    out << "  Reduced edges: " << result.reducedEdges << ", seeds: " << result.seedsDirect << " direct / "
        << result.seedsStructural << " structural, rank changes: " << result.rankChanges << "\n";
    if (!result.checked) return;
    out << "  Full-customization diff: " << result.comparison.mismatches << " mismatch(es) over "
        << result.comparison.edgesCompared << " edges (under " << result.comparison.underApproximated << ", over "
        << result.comparison.overApproximated << ")\n";
}


/**
 * Build a cell index, print a per-level profile, and validate it against the timetable.
 *
 * The two indices expose the same `numberOfLevels()/cellsAtLevel()/byteSize()/validate()` shape,
 * so their shell commands differ only in the per-level statistic they report -- which is what
 * `summary` and `perLevel` supply.
 */
template <typename Index, typename Summary, typename PerLevel>
inline void buildAndReportCellIndex(const std::string& name, const DynamicTimeTable::Data& data,
                                    const Summary& summary, const PerLevel& perLevel) {
    Timer timer;
    Index index(data);
    const double buildMs = timer.elapsedMilliseconds();

    std::cout << name << ": " << index.numberOfLevels() << " levels, " << summary(index) << ", "
              << String::bytesToString(index.byteSize()) << ", built in " << buildMs << " ms." << std::endl;
    for (int level = 0; level < index.numberOfLevels(); ++level) {
        std::cout << "  level " << level << ": " << perLevel(index, level) << std::endl;
    }

    std::string error;
    if (index.validate(data, error)) {
        std::cout << "Validation passed." << std::endl;
    } else {
        std::cout << "Validation FAILED: " << error << std::endl;
    }
}

/**
 * One simulated minute: generate an update, apply it to timetable and transfers, then customize.
 *
 * Shared by the one-shot command and the timeline so that the per-minute pipeline has exactly one
 * spelling -- a new phase or a changed `customizeStep` signature is then a single edit.
 */
struct CustomizationStep {
    CustomizationStepResult result;
    PhaseTimings phases{};
    DynamicTimeTable::Algo::UpdateSimulationStats simStats{};
    DynamicTimeTable::PendingUpdates updates;
    TimedAppliedUpdate applied;
};

inline CustomizationStep runCustomizationStep(DynamicTimeTable::Data& dynamicTimeTable, CustomizationHarness& harness,
                                              const DynamicTimeTable::Algo::UpdateSimulationConfig& config,
                                              const int now, const int threads, const bool check) {
    PhaseTimings phases{};
    DynamicTimeTable::Algo::UpdateSimulationStats simStats{};
    DynamicTimeTable::PendingUpdates updates = UpdateGenerator(config)(dynamicTimeTable, now, &simStats, &phases);

    TimedAppliedUpdate applied = applyIncrementalUpdateTimed(dynamicTimeTable, harness.updater(), updates, threads,
                                                             CustomizingTransferUpdater::noTimeCutoff);
    phases += applied.phases;

    CustomizationStepResult result = harness.customizeStep(dynamicTimeTable, applied.applied.queryData,
                                                           *applied.applied.changes, phases, check);
    return CustomizationStep{std::move(result), phases, simStats, std::move(updates), std::move(applied)};
}

}  // namespace DynamicCustomizationCommands

class BuildCellBorderIndex : public ParameterizedCommand {
public:
    BuildCellBorderIndex(BasicShell& shell)
        : ParameterizedCommand(shell, "buildCellBorderIndex",
                               "Builds the TREX cell-border index for a partitioned DynamicTimeTable, validates it "
                               "against the timetable, and prints per-level border counts.") {
        addParameter("Input binary (DynamicTimeTable Data)");
    }

    virtual void execute() noexcept override {
        using namespace DynamicCustomizationCommands;

        const auto dynamicTimeTable =
            loadPartitionedDynamicTimeTable(getParameter("Input binary (DynamicTimeTable Data)"));
        if (!dynamicTimeTable) return;

        buildAndReportCellIndex<DynamicTB::Customization::CellBorderIndex>(
            "Cell border index", *dynamicTimeTable,
            [](const auto& index) { return std::to_string(index.positionCount()) + " filed positions"; },
            [](const auto& index, const int level) {
                std::size_t positions = 0;
                std::size_t nonEmptyCells = 0;
                for (std::size_t cell = 0; cell < index.cellsAtLevel(level); ++cell) {
                    const std::size_t size = index.positions(level, static_cast<std::uint16_t>(cell)).size();
                    positions += size;
                    if (size > 0) ++nonEmptyCells;
                }
                return std::to_string(positions) + " positions over " + std::to_string(nonEmptyCells) + " / " +
                       std::to_string(index.cellsAtLevel(level)) + " cells";
            });
    }
};

class BuildCellStopIndex : public ParameterizedCommand {
public:
    BuildCellStopIndex(BasicShell& shell)
        : ParameterizedCommand(shell, "buildCellStopIndex",
                               "Builds the cell->stops index for a partitioned DynamicTimeTable, validates it "
                               "against the timetable, and prints a per-level size profile.") {
        addParameter("Input binary (DynamicTimeTable Data)");
    }

    virtual void execute() noexcept override {
        using namespace DynamicCustomizationCommands;

        const auto dynamicTimeTable =
            loadPartitionedDynamicTimeTable(getParameter("Input binary (DynamicTimeTable Data)"));
        if (!dynamicTimeTable) return;

        buildAndReportCellIndex<CellStopIndex>(
            "Cell stop index", *dynamicTimeTable,
            [](const auto& index) { return std::to_string(index.stopCount()) + " stops"; },
            [](const auto& index, const int level) {
                std::size_t nonEmpty = 0;
                std::size_t largest = 0;
                for (std::size_t cell = 0; cell < index.cellsAtLevel(level); ++cell) {
                    const std::size_t size = index.stops(level, static_cast<std::uint16_t>(cell)).size();
                    if (size > 0) ++nonEmpty;
                    largest = std::max(largest, size);
                }
                return std::to_string(nonEmpty) + " / " + std::to_string(index.cellsAtLevel(level)) +
                       " non-empty cells, largest holds " + std::to_string(largest) + " stops";
            });
    }
};

/**
 * The full customization, end to end: build the store, export the reduced CSR, customize every
 * cell at every level, and (optionally) round-trip the result through the persistent store to
 * check that the sparse, set-semantics write-back reproduces the sweep's ranks exactly.
 */
class CustomizeDynamic : public DynamicTransferCommandBase {
public:
    CustomizeDynamic(BasicShell& shell)
        : DynamicTransferCommandBase(shell, "customizeDynamic",
                                     "Runs a full TREX customization over the reduced transfer export of a "
                                     "partitioned DynamicTimeTable.") {
        addParameter("Input binary (DynamicTimeTable Data)");
        addParameter("Number of threads", "max");
        addParameter("Verify rank write-back", "true");
    }

    virtual void execute() noexcept override {
        using namespace DynamicCustomizationCommands;

        const int threads = numberOfThreads();
        const bool verifyWriteBack = getParameter("Verify rank write-back") != "false";

        const auto loaded = loadPartitionedDynamicTimeTable(getParameter("Input binary (DynamicTimeTable Data)"));
        if (!loaded) return;
        const DynamicTimeTable::Data& dynamicTimeTable = *loaded;
        std::cout << "Partition: " << dynamicTimeTable.getNumberOfLevels() << " levels." << std::endl;

        const DynamicQueryData queryData = QueryDataBuilder()(dynamicTimeTable);
        TransferStoreType store;
        TransferUpdater updater(store);
        TripBased::Transfers csr = buildReducedExport(updater, queryData, threads);

        Timer timer;
        CellIndices indices;
        indices.build(dynamicTimeTable);
        std::cout << "Cell indices built in " << timer.elapsedMilliseconds() << " ms ("
                  << String::bytesToString(indices.byteSize()) << ")" << std::endl;

        const CellNetwork net = indices.bind(dynamicTimeTable, queryData, csr);
        FullCustomizationDriver driver(threads);

        timer.restart();
        driver.run(net);
        const double sweepMs = timer.elapsedMilliseconds();

        std::cout << "Full sweep finished in " << sweepMs << " ms on " << threads << " thread(s), "
                  << driver.changes().size() << " rank changes, scratch "
                  << String::bytesToString(driver.scratchByteSize()) << std::endl;
        printCsrRankHistogram(csr);
        printCustomizationStats(driver.stats().levels(), std::cout);

        if (!verifyWriteBack) return;

        // Round-trip: push the decisions into the persistent store, re-export, and require the
        // ranks to come back unchanged. This is what proves the sparse write-back is complete --
        // a full sweep decides every edge, so anything the write-back drops shows up here.
        const std::vector<std::uint8_t> expected = snapshotRanks(csr);
        timer.restart();
        updater.applyRankChanges(driver.changes(), threads);
        std::cout << "Rank write-back of " << driver.changes().size() << " changes in "
                  << timer.elapsedMilliseconds() << " ms" << std::endl;

        TripBased::Transfers reExported = updater.exportReducedTransfers(queryData, threads);
        const RankComparison comparison = compareRankSnapshot(reExported, expected);
        std::cout << "Write-back round trip: " << (comparison.identical() ? "PASSED" : "FAILED") << std::endl;
        if (!comparison.identical()) printRankMismatches(comparison);
    }
};

/**
 * The cross-check against the static TREX builder.
 *
 * The full customization shares its kernel with the incremental one by design, so a
 * full-vs-incremental diff cannot see a bug that lives *in* the kernel. This command is the only
 * check that can: it compares the dynamic full customization against the ranks the static TREX
 * `Builder` wrote into `trex.binary`'s `LocalLevel`, which is an independent implementation on
 * an independent data structure.
 *
 * Two preconditions, both verified here rather than assumed:
 *
 *  1. Both instances must carry the *same* partition. Different cell ids make a rank comparison
 *     meaningless, not merely noisy.
 *  2. Topology first, ranks second. The dynamic reduced export must equal the static
 *     `stopEventGraph` as an edge set; a rank mismatch on a differing edge set proves nothing.
 *     Both minimizers order equal-arrival candidates by target stop event before folding (see the
 *     tie-break in `Algorithms/TripBased/Preprocessing/StopEventGraphBuilder.h`), so the two
 *     reduced sets agree exactly and this precondition holds by construction. If a topology
 *     difference ever appears first analyse that.
 *
 * The static side must be built with route-based pruning *off*
 * (`raptorToTREX <raptor.binary> <out> <levels> false max 1`); `raptorToTREX` defaults it on,
 * which the dynamic generator does not do.
 */
class CompareDynamicCustomizationToStatic : public DynamicTransferCommandBase {
public:
    CompareDynamicCustomizationToStatic(BasicShell& shell)
        : DynamicTransferCommandBase(shell, "compareDynamicCustomizationToStatic",
                                     "Compares the dynamic full customization against the LocalLevel ranks stored "
                                     "in a matching static TREX instance, edge by edge.") {
        addParameter("Input binary (DynamicTimeTable Data)");
        addParameter("Input binary (TREX Data)");
        addParameter("Number of threads", "max");
    }

    virtual void execute() noexcept override {
        using namespace DynamicCustomizationCommands;

        const int threads = numberOfThreads();

        const auto loaded = loadPartitionedDynamicTimeTable(getParameter("Input binary (DynamicTimeTable Data)"));
        if (!loaded) return;
        const DynamicTimeTable::Data& dynamicTimeTable = *loaded;

        std::cout << "Loading static TREX instance..." << std::endl;
        TripBased::TREXData trexData(getParameter("Input binary (TREX Data)"));

        if (!partitionsAgree(dynamicTimeTable, trexData)) return;

        const DynamicQueryData queryData = QueryDataBuilder()(dynamicTimeTable);
        TransferStoreType store;
        TransferUpdater updater(store);
        TripBased::Transfers csr = buildReducedExport(updater, queryData, threads);

        CellIndices indices;
        indices.build(dynamicTimeTable);
        const CellNetwork net = indices.bind(dynamicTimeTable, queryData, csr);

        Timer timer;
        FullCustomizationDriver driver(threads);
        driver.run(net, DynamicTB::Customization::ChangeOutput::Discard);
        std::cout << "Full sweep finished in " << timer.elapsedMilliseconds() << " ms." << std::endl;
        printCsrRankHistogram(csr);

        const std::vector<RankedEdge> dynamicEdges = extractRankedEdges(csr);
        const std::vector<RankedEdge> staticEdges = extractStaticRankedEdges(trexData);
        // The aggregate distributions separate "the two kernels disagree about what a rank means"
        // from "the two kernels agree but are looking at slightly different graphs".
        printRankedEdgeHistogram("Static rank distribution", staticEdges);
        const EndpointRankComparison comparison = compareRanksByEndpoint(dynamicEdges, staticEdges);

        std::cout << "\nTopology" << std::endl;
        std::cout << "  Dynamic reduced edges: " << dynamicEdges.size() << std::endl;
        std::cout << "  Static stopEventGraph edges: " << staticEdges.size() << std::endl;
        std::cout << "  Only in dynamic: " << comparison.edgesOnlyInLeft << std::endl;
        std::cout << "  Only in static: " << comparison.edgesOnlyInRight << std::endl;
        if (comparison.edgesOnlyInLeft != 0 || comparison.edgesOnlyInRight != 0) {
            std::cout << "  Edge sets differ -- the rank comparison below is only over the shared edges and "
                         "does not settle whether the kernels agree. Check that the static instance was built "
                         "without route-based pruning, and that both minimizers use the same tie-break."
                      << std::endl;
        }

        std::cout << "\nRanks" << std::endl;
        printRankMismatches(comparison.ranks);
        std::cout << "Static cross-check (native topologies): " << (comparison.identical() ? "PASSED" : "FAILED")
                  << std::endl;
    }

private:
    /// Same stop count and the same cell id for every stop; anything else makes ranks incomparable.
    static bool partitionsAgree(const DynamicTimeTable::Data& dynamicTimeTable, const TripBased::TREXData& trexData) {
        if (dynamicTimeTable.getNumberOfLevels() != trexData.getNumberOfLevels()) {
            std::cout << "Level counts differ: dynamic has " << dynamicTimeTable.getNumberOfLevels()
                      << ", static has " << trexData.getNumberOfLevels() << "." << std::endl;
            return false;
        }
        if (dynamicTimeTable.numberOfStops() != trexData.numberOfStops()) {
            std::cout << "Stop counts differ: dynamic has " << dynamicTimeTable.numberOfStops() << ", static has "
                      << trexData.numberOfStops() << "." << std::endl;
            return false;
        }
        for (StopId stop(0); stop < StopId(dynamicTimeTable.numberOfStops()); ++stop) {
            if (dynamicTimeTable.getCellIdOfStop(stop) != trexData.getCellIdOfStop(stop)) {
                std::cout << "Cell ids differ at stop " << stop << " (dynamic "
                          << dynamicTimeTable.getCellIdOfStop(stop) << ", static " << trexData.getCellIdOfStop(stop)
                          << ") -- apply the same partition file to both instances." << std::endl;
                return false;
            }
        }
        std::cout << "Partitions agree: " << dynamicTimeTable.getNumberOfLevels() << " levels over "
                  << dynamicTimeTable.numberOfStops() << " stops." << std::endl;
        return true;
    }

    /// The static side's ranked edge set: `stopEventGraph` plus its `LocalLevel` attribute.
    static std::vector<DynamicTB::Customization::RankedEdge> extractStaticRankedEdges(
        const TripBased::TREXData& trexData) {
        std::vector<DynamicTB::Customization::RankedEdge> edges;
        edges.reserve(trexData.stopEventGraph.numEdges());
        for (const auto [edge, from] : trexData.stopEventGraph.edgesWithFromVertex()) {
            edges.push_back({static_cast<std::uint32_t>(from),
                             static_cast<std::uint32_t>(trexData.stopEventGraph.get(ToVertex, edge)),
                             static_cast<std::uint8_t>(trexData.stopEventGraph.get(LocalLevel, edge))});
        }
        std::sort(edges.begin(), edges.end());
        return edges;
    }
};

/**
 * One update, incremental customization vs. a full one on the same export.
 *
 * The shape of the check matters: both sides run on the *same* CSR, so the diff is positional and
 * cannot be confused by a topology change, and the incremental side starts from a store that a
 * full customization has already been written back into. What it measures is therefore exactly
 * the claim the incremental customization makes -- that customizing only the disturbed cells
 * reproduces the exact ranks -- and nothing else.
 */
class SimulateAndCompareCustomization : public DynamicTransferCommandBase {
public:
    SimulateAndCompareCustomization(BasicShell& shell)
        : DynamicTransferCommandBase(
              shell, "simulateAndCompareCustomization",
              "Customizes a partitioned DynamicTimeTable, applies one simulated update, customizes it "
              "incrementally, and diffs the result against a full customization edge by edge.") {
        addParameter("Input binary (DynamicTimeTable Data)");
        addParameter("Current time (seconds)", "28800");
        addSimulationParameters("42", "5", "50", "20");
        addParameter("Number of threads", "max");
    }

    virtual void execute() noexcept override {
        using namespace DynamicCustomizationCommands;

        const int threads = numberOfThreads();
        const int now = getParameter<int>("Current time (seconds)");

        auto loaded = loadPartitionedDynamicTimeTable(getParameter("Input binary (DynamicTimeTable Data)"));
        if (!loaded) return;
        DynamicTimeTable::Data& dynamicTimeTable = *loaded;
        std::cout << "Partition: " << dynamicTimeTable.getNumberOfLevels() << " levels." << std::endl;

        const DynamicQueryData initialQueryData = QueryDataBuilder()(dynamicTimeTable);
        CustomizationHarness harness(threads, /* keepReference = */ true);
        harness.buildInitialState(dynamicTimeTable, initialQueryData);

        const CustomizationStep step =
            runCustomizationStep(dynamicTimeTable, harness, simulationConfig(), now, threads, /* check = */ true);

        printSimulationSummary(step.simStats, step.updates, step.applied.applied.statistics,
                               *step.applied.applied.changes);
        std::cout << "\nCustomization" << std::endl;
        printStepResult(step.result, std::cout);
        printCustomizationStats(harness.incremental().stats().levels(), std::cout);
        printPhaseTimings(step.phases);
        if (!step.result.ok()) printRankMismatches(step.result.comparison);
        std::cout << "\nIncremental customization vs. full customization: "
                  << (step.result.ok() ? "PASSED" : "FAILED") << std::endl;
    }
};

/**
 * The minute-by-minute Dynamic TREX loop: update the timetable, update the transfers, customize
 * the ranks incrementally, write them back -- and, in the checking variant, prove each minute's
 * result equals a full customization of the same export.
 */
class SimulateCustomizationTimelineBase : public DynamicTransferCommandBase {
protected:
    SimulateCustomizationTimelineBase(BasicShell& shell, const std::string& name, const std::string& description,
                                      const bool check)
        : DynamicTransferCommandBase(shell, name, description), check_(check) {
        addParameter("Input binary (DynamicTimeTable Data)");
        addParameter("Start time (seconds)", "28800");
        addParameter("End time (seconds)", "32400");
        addParameter("Step seconds", "60");
        addParameter("Base Random seed", "1");
        addParameter("Timing CSV output path", "");
        addSimulationRateParameters("5", "50", "20");
        addParameter("Number of threads", "max");
    }

    void runTimeline() {
        using namespace DynamicCustomizationCommands;

        const int threads = numberOfThreads();
        const int startTimeSeconds = getParameter<int>("Start time (seconds)");
        const int endTimeSeconds = getParameter<int>("End time (seconds)");
        const int stepSeconds = std::max(1, getParameter<int>("Step seconds"));
        const uint32_t baseSeed = static_cast<uint32_t>(getParameter<int>("Base Random seed"));

        auto loaded = loadPartitionedDynamicTimeTable(getParameter("Input binary (DynamicTimeTable Data)"));
        if (!loaded) return;
        DynamicTimeTable::Data& dynamicTimeTable = *loaded;
        std::cout << "Partition: " << dynamicTimeTable.getNumberOfLevels() << " levels." << std::endl;

        std::ofstream timingCsv;
        const std::string timingCsvPath = getParameter("Timing CSV output path");
        if (!timingCsvPath.empty()) {
            timingCsv.open(timingCsvPath);
            if (!timingCsv) {
                std::cerr << "Failed to open timing CSV file: " << timingCsvPath << std::endl;
                return;
            }
            writeCombinedCsvHeader(timingCsv);
        }

        const DynamicQueryData initialQueryData = QueryDataBuilder()(dynamicTimeTable);
        CustomizationHarness harness(threads, /* keepReference = */ check_);
        harness.buildInitialState(dynamicTimeTable, initialQueryData);

        PhaseTimingsAccumulator timingAccumulator;
        std::vector<std::pair<int, CustomizationStepResult>> divergences;
        std::size_t totalRankChanges = 0;
        int stepIndex = 0;

        for (int now = startTimeSeconds; now < endTimeSeconds; now += stepSeconds, ++stepIndex) {
            const uint32_t seed = baseSeed + static_cast<uint32_t>(stepIndex);
            std::cout << "\rTimeline step " << (stepIndex + 1) << " at t=" << now << "s (seed " << seed << ")..."
                      << std::flush;

            const CustomizationStep step = runCustomizationStep(dynamicTimeTable, harness,
                                                               simulationConfigWithSeed(seed), now, threads, check_);
            totalRankChanges += step.result.rankChanges;
            if (!step.result.ok()) {
                divergences.emplace_back(now, step.result);
                std::cout << "\n!!! RANK DIVERGENCE at t=" << now << "s (seed " << seed << ") !!!" << std::endl;
                printStepResult(step.result, std::cout);
                printRankMismatches(step.result.comparison);
            }

            timingAccumulator.add(step.phases);
            if (timingCsv) {
                const UpdateMemoryStats memory =
                    collectMemoryStats(harness.store(), step.applied.applied.queryData, dynamicTimeTable);
                writeCombinedCsvRow(stepIndex + 1, step.phases, step.applied.counters, memory, timingCsv);
            }
        }

        std::cout << "\nTimeline finished: " << stepIndex << " step(s), " << totalRankChanges
                  << " rank changes in total." << std::endl;
        if (check_) {
            std::cout << (divergences.empty() ? "Incremental customization matched the full customization at every "
                                                "step."
                                              : "Incremental customization DIVERGED:")
                      << std::endl;
            for (const auto& [t, result] : divergences) {
                std::cout << "  t=" << t << "s: " << result.comparison.mismatches << " mismatch(es), "
                          << result.comparison.underApproximated << " under-approximated" << std::endl;
            }
        }
        printPhaseTimingsSummary(timingAccumulator);
        // Per-level stats are reset at the start of every run, so what survives here describes the
        // last step only -- useful as a shape, not as a total.
        std::cout << "\nPer-level customization stats of the last step" << std::endl;
        printCustomizationStats(harness.incremental().stats().levels(), std::cout);
    }

private:
    bool check_;
};

class SimulateAndCompareCustomizationTimeline : public SimulateCustomizationTimelineBase {
public:
    SimulateAndCompareCustomizationTimeline(BasicShell& shell)
        : SimulateCustomizationTimelineBase(
              shell, "simulateAndCompareCustomizationTimeline",
              "Minute-by-minute incremental customization, each step diffed against a full customization "
              "of the same export.",
              /* check = */ true) {}

    virtual void execute() noexcept override { runTimeline(); }
};

class SimulateCustomizationTimeline : public SimulateCustomizationTimelineBase {
public:
    SimulateCustomizationTimeline(BasicShell& shell)
        : SimulateCustomizationTimelineBase(shell, "simulateCustomizationTimeline",
                                            "Minute-by-minute incremental customization without checking -- the "
                                            "measurement run.",
                                            /* check = */ false) {}

    virtual void execute() noexcept override { runTimeline(); }
};
