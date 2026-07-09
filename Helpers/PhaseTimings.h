#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string_view>

// Per-phase wall-clock timing breakdown for one applied update, used to profile the
// update pipeline under multi-threaded execution. Each field is a single aggregate
// duration for that phase (timed at the sequential call site wrapping the phase, not a
// per-thread breakdown), so no OpenMP-awareness is needed here.
struct PhaseTimings {
    std::chrono::microseconds updateGeneration{0};
    std::chrono::microseconds timetableUpdate{0};
    std::chrono::microseconds queryDataExport{0};
    std::chrono::microseconds baseTransferUpdate{0};
    std::chrono::microseconds minimizationUpdate{0};
    std::chrono::microseconds exportPhase{0};

    PhaseTimings& operator+=(const PhaseTimings& other) noexcept {
        updateGeneration += other.updateGeneration;
        timetableUpdate += other.timetableUpdate;
        queryDataExport += other.queryDataExport;
        baseTransferUpdate += other.baseTransferUpdate;
        minimizationUpdate += other.minimizationUpdate;
        exportPhase += other.exportPhase;
        return *this;
    }

    std::chrono::microseconds total() const noexcept {
        return updateGeneration + timetableUpdate + queryDataExport + baseTransferUpdate + minimizationUpdate +
               exportPhase;
    }

    // Element-wise minimum/maximum, used to track spread across iterations/timeline steps.
    static PhaseTimings min(const PhaseTimings& a, const PhaseTimings& b) noexcept {
        return {std::min(a.updateGeneration, b.updateGeneration), std::min(a.timetableUpdate, b.timetableUpdate),
                std::min(a.queryDataExport, b.queryDataExport), std::min(a.baseTransferUpdate, b.baseTransferUpdate),
                std::min(a.minimizationUpdate, b.minimizationUpdate), std::min(a.exportPhase, b.exportPhase)};
    }

    static PhaseTimings max(const PhaseTimings& a, const PhaseTimings& b) noexcept {
        return {std::max(a.updateGeneration, b.updateGeneration), std::max(a.timetableUpdate, b.timetableUpdate),
                std::max(a.queryDataExport, b.queryDataExport), std::max(a.baseTransferUpdate, b.baseTransferUpdate),
                std::max(a.minimizationUpdate, b.minimizationUpdate), std::max(a.exportPhase, b.exportPhase)};
    }
};

// Accumulates PhaseTimings across many iterations/timeline steps, tracking sum, min, and
// max per phase for an end-of-run human-readable summary.
class PhaseTimingsAccumulator {
public:
    void add(const PhaseTimings& sample) noexcept {
        if (count == 0) {
            minSample = sample;
            maxSample = sample;
        } else {
            minSample = PhaseTimings::min(minSample, sample);
            maxSample = PhaseTimings::max(maxSample, sample);
        }
        sum += sample;
        ++count;
    }

    std::size_t sampleCount() const noexcept { return count; }
    const PhaseTimings& sumTotals() const noexcept { return sum; }
    const PhaseTimings& minTotals() const noexcept { return minSample; }
    const PhaseTimings& maxTotals() const noexcept { return maxSample; }

    PhaseTimings meanTotals() const noexcept {
        if (count == 0) return {};
        PhaseTimings mean;
        mean.updateGeneration = sum.updateGeneration / static_cast<long>(count);
        mean.timetableUpdate = sum.timetableUpdate / static_cast<long>(count);
        mean.queryDataExport = sum.queryDataExport / static_cast<long>(count);
        mean.baseTransferUpdate = sum.baseTransferUpdate / static_cast<long>(count);
        mean.minimizationUpdate = sum.minimizationUpdate / static_cast<long>(count);
        mean.exportPhase = sum.exportPhase / static_cast<long>(count);
        return mean;
    }

private:
    PhaseTimings sum{};
    PhaseTimings minSample{};
    PhaseTimings maxSample{};
    std::size_t count{0};
};

inline void printPhaseTimings(const PhaseTimings& t, std::ostream& out = std::cout) {
    const auto total = t.total();
    const double totalUs = static_cast<double>(total.count());
    auto printRow = [&](std::string_view name, std::chrono::microseconds value) {
        const double pct = totalUs > 0.0 ? (static_cast<double>(value.count()) / totalUs) * 100.0 : 0.0;
        out << "  " << std::left << std::setw(22) << name << std::right << std::setw(12) << value.count() << " us  "
            << std::fixed << std::setprecision(1) << pct << "%\n";
    };
    out << "Phase timing breakdown\n";
    printRow("Update generation", t.updateGeneration);
    printRow("Timetable update", t.timetableUpdate);
    printRow("Query data export", t.queryDataExport);
    printRow("Base transfer update", t.baseTransferUpdate);
    printRow("Minimization update", t.minimizationUpdate);
    printRow("Export", t.exportPhase);
    out << "  " << std::left << std::setw(22) << "Total" << std::right << std::setw(12) << total.count() << " us\n";
}

inline void printPhaseTimingsSummary(const PhaseTimingsAccumulator& acc, std::ostream& out = std::cout) {
    out << "Phase timing summary over " << acc.sampleCount() << " sample(s)\n";
    out << "-- Sum --\n";
    printPhaseTimings(acc.sumTotals(), out);
    out << "-- Mean --\n";
    printPhaseTimings(acc.meanTotals(), out);
    out << "-- Min --\n";
    printPhaseTimings(acc.minTotals(), out);
    out << "-- Max --\n";
    printPhaseTimings(acc.maxTotals(), out);
}

inline void writePhaseTimingsCsvHeader(std::ostream& out) {
    out << "index,updateGeneration_us,timetableUpdate_us,queryDataExport_us,baseTransferUpdate_us,"
           "minimizationUpdate_us,export_us,total_us\n";
}

inline void writePhaseTimingsCsvRow(const PhaseTimings& t, const long index, std::ostream& out) {
    out << index << ',' << t.updateGeneration.count() << ',' << t.timetableUpdate.count() << ','
        << t.queryDataExport.count() << ',' << t.baseTransferUpdate.count() << ',' << t.minimizationUpdate.count()
        << ',' << t.exportPhase.count() << ',' << t.total().count() << '\n';
}
