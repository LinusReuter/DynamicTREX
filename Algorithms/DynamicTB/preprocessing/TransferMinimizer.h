#pragma once

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#include "../../../Helpers/MultiThreading.h"
#include "../../../Helpers/Types.h"
#include "../../../Helpers/UpdateCounters.h"
#include "../../DynamicTimeTable/BuildQueryData.h"
#include "AffectedEventSink.h"
#include "TransferTypes.h"

namespace DynamicTB::Preprocessing {

/**
 * @brief Per-thread scratch for minimization.
 *
 * `labels` is one StopLabel per stop and is reset in O(1) via the generation counter
 * `timestamp` (see StopLabel::checkTimestamp), so it is allocated once per thread and
 * reused for every trip that thread reduces.
 */
struct MinimizationWorkspace {
    std::vector<StopLabel> labels;
    int timestamp{0};
};

/**
 * @brief Computes the reduced (minimized) transfer set.
 *
 * The store always holds the FULL set; minimization only decides `TransferMeta::isMinimized`
 * per edge. Every flip of that flag is reported to the AffectedSink -- that delta is exactly
 * the "directly affected on level 0" set Dynamic TREX customization starts from.
 */
template <class Store, class AffectedSink = NullAffectedSink>
class TransferMinimizer {
public:
    using DynamicQueryData = DynamicTimeTable::Algo::DynamicQueryData;
    using NodeID = PersistentStopEventId;
    using OutEdge = typename Store::OutEdge;

    TransferMinimizer(Store& store, const DynamicQueryData& queryData) noexcept
        : store_(store), queryData_(&queryData) {}

    inline void setQueryData(const DynamicQueryData& queryData) noexcept { queryData_ = &queryData; }

    /**
     * @brief Parallel driver: re-run minimization for the given (flat) trips.
     * Each thread owns its scratch buffers, its counters and its affected-set collector.
     *
     * `sinkFactory()` must return a fresh per-thread sink; `mergeSink(sink)` is called under
     * an `omp critical` to fold it into the shared result.
     */
    template <class SinkFactory, class SinkMerge>
    void run(std::span<const MinTarget> trips, const int numberOfThreads, const int nowSeconds,
             TransferUpdateCounters* counters, SinkFactory&& sinkFactory, SinkMerge&& mergeSink) const {
        const std::size_t numStops = queryData_->queryData.firstRouteSegmentOfStop.size() - 1;

        const int threads = std::max(1, numberOfThreads);
        omp_set_num_threads(threads);
#pragma omp parallel if (threads > 1)
        {
            if (omp_get_num_threads() > 1) pinThreadToCoreId(omp_get_thread_num() % numberOfCores());
            MinimizationWorkspace ws;
            ws.labels.assign(numStops, StopLabel());
            std::vector<TransferCandidate> localCandidates;
            TransferUpdateCounters localCtr;
            auto localSink = sinkFactory();
#pragma omp for schedule(dynamic, 1)
            for (const auto& [trip, startIndex] : trips) {
                reduceTransfersForTrip(trip, static_cast<int>(startIndex), ws, localCandidates, nowSeconds, localCtr,
                                       localSink);
            }
#pragma omp critical
            {
                mergeSink(localSink);
                if constexpr (collectTransferStats) {
                    if (counters != nullptr) *counters += localCtr;
                }
            }
        }
    }

    /**
     * @brief Candidate outgoing transfer during minimization.
     * Holds a pointer directly into the store's outgoing edge so the keep-flag can
     * be written in place (O(1))
     */
    struct TransferCandidate {
        OutEdge* edge;
        StopEventId flatTo;
        int destArrivalTime;
    };

    /**
     * @brief Folds one candidate's domination profile into the workspace labels.
     * @return true if the candidate improved at least one label (i.e. it is kept).
     * A non-improving candidate writes NO labels, so replaying only kept edges reproduces the
     * exact same label profile regardless of order (all updates are min).
     */
    inline bool foldCandidateDomination(const TransferCandidate& candidate, const int localTimestamp,
                                        std::vector<StopLabel>& localLabels) const {
        const auto& qd = queryData_->queryData;
        bool keep = false;
        const TripId toTrip = queryData_->tripOfEvent(candidate.flatTo);
        const StopEventId firstEventOfToTrip = qd.firstStopEventOfTrip[toTrip];
        const size_t numStopsInToTrip = qd.firstStopEventOfTrip[toTrip + 1] - firstEventOfToTrip;

        for (size_t j = numStopsInToTrip - static_cast<size_t>(StopIndex(candidate.flatTo - firstEventOfToTrip)) - 1;
             j > 0; --j) {
            StopEventId destEvent = StopEventId(candidate.flatTo + j);
            StopId destinationStop = queryData_->stopOfEvent(destEvent);
            int destinationArrivalTime = static_cast<int>(queryData_->arrivalTimeOfEvent(destEvent));

            localLabels[destinationStop].checkTimestamp(localTimestamp);
            if (localLabels[destinationStop].arrivalTime > destinationArrivalTime) {
                localLabels[destinationStop].arrivalTime = destinationArrivalTime;
                keep = true;
            }

            for (const auto edge : qd.transferGraph.edgesFrom(destinationStop)) {
                StopId arrivalStop = StopId(qd.transferGraph.get(ToVertex, edge));
                int arrivalTimeAtStop = destinationArrivalTime + qd.transferGraph.get(TravelTime, edge);

                localLabels[arrivalStop].checkTimestamp(localTimestamp);
                if (localLabels[arrivalStop].arrivalTime > arrivalTimeAtStop) {
                    localLabels[arrivalStop].arrivalTime = arrivalTimeAtStop;
                    keep = true;
                }
            }
        }
        return keep;
    }

    /**
     * @brief Evaluates and assigns minimization metadata for transfers of a given trip.
     *
     * Warm-start: keep-flags for stops with index > startIndex are already correct (nothing above
     * changed), so those stops are not re-decided -- we only replay their already-kept edges to
     * rebuild the StopLabel profile (no sort, no non-kept scan, no writes). Full re-decision runs
     * for stops in [1, startIndex]. Because that region writes no flags, it also emits no
     * affected-set entries.
     *
     * Time cutoff (compile-gated): stop events are time-ordered, so once a stop's arrival is in the
     * past (< nowSeconds) so is every earlier stop; we break and keep their existing flags. This is
     * only correct for queries starting at/after nowSeconds (past-boarding transfers keep stale
     * flags), hence opt-in.
     */
    void reduceTransfersForTrip(TripId flatTrip, const int startIndex, MinimizationWorkspace& ws,
                                std::vector<TransferCandidate>& candidates,
                                [[maybe_unused]] const int nowSeconds, TransferUpdateCounters& lc,
                                AffectedSink& sink) const {
        const auto& qd = queryData_->queryData;
        std::vector<StopLabel>& localLabels = ws.labels;
        const int localTimestamp = ++ws.timestamp;

        int numStops = qd.firstStopEventOfTrip[flatTrip + 1] - qd.firstStopEventOfTrip[flatTrip];

        // Scan backward from the destination stop down to the second stop (index 1)
        for (int i = numStops - 1; i > 0; --i) {
            if constexpr (collectTransferStats) ++lc.minStopsScanned;
            StopEventId flatFromEvent = StopEventId(qd.firstStopEventOfTrip[flatTrip] + i);
            PersistentStopEventId pFromEvent = queryData_->flatToPersistentEvent[flatFromEvent];
            assert(pFromEvent != noPersistentStopEventId);

            int arrivalTime = static_cast<int>(queryData_->arrivalTimeOfEvent(flatFromEvent));
            StopId fromStop = queryData_->stopOfEvent(flatFromEvent);

#ifdef DYN_TRANSFER_TIME_CUTOFF
            // Time cutoff: this stop and all earlier ones are in the past; keep their flags as-is.
            if (arrivalTime < nowSeconds) break;
#endif

            // 1. Update labels for the stop itself and its outgoing transfer/footpath neighbors
            localLabels[fromStop].update(localTimestamp, arrivalTime);
            for (const auto edge : qd.transferGraph.edgesFrom(fromStop)) {
                auto toStop = StopId(qd.transferGraph.get(ToVertex, edge));
                const int transferTime = qd.transferGraph.get(TravelTime, edge);
                localLabels[toStop].update(localTimestamp, arrivalTime + transferTime);
            }

            // Warm-start region: only rebuild labels from already-kept edges, don't re-decide.
            if (i > startIndex) {
                for (auto& edge : store_.outgoing_mutable(NodeID(pFromEvent))) {
                    if (!edge.meta.isMinimized) continue;
                    StopEventId flatTo = queryData_->persistentToFlatEvent[edge.to];
                    if (flatTo == noStopEvent) continue;
                    if constexpr (collectTransferStats) ++lc.minWarmStartReplays;
                    foldCandidateDomination({&edge, flatTo, static_cast<int>(queryData_->arrivalTimeOfEvent(flatTo))},
                                            localTimestamp, localLabels);
                }
                continue;
            }

            // 2. Gather full unreduced candidates from the edge store.
            candidates.clear();
            for (auto& edge : store_.outgoing_mutable(NodeID(pFromEvent))) {
                StopEventId flatTo = queryData_->persistentToFlatEvent[edge.to];
                if (flatTo != noStopEvent) {
                    candidates.push_back({&edge, flatTo, static_cast<int>(queryData_->arrivalTimeOfEvent(flatTo))});
                }
            }
            if constexpr (collectTransferStats) lc.minCandidatesEvaluated += candidates.size();

            // 3. Sort candidates by destination arrival time.
            std::ranges::sort(candidates, [](const TransferCandidate& a, const TransferCandidate& b) {
                if (a.destArrivalTime != b.destArrivalTime) return a.destArrivalTime < b.destArrivalTime;
                return a.edge->to < b.edge->to;
            });

            // 4. Domination profile filtering
            for (const auto& candidate : candidates) {
                const bool wasMinimized = candidate.edge->meta.isMinimized;
                const bool keep = foldCandidateDomination(candidate, localTimestamp, localLabels);
                candidate.edge->meta.isMinimized = keep;
                if (keep != wasMinimized) {
                    // The reduced set of both endpoints changed => level-0 affected (TREX).
                    sink.markEdgeChanged(pFromEvent, candidate.edge->to);
                    if constexpr (collectTransferStats) ++lc.minimizationFlips;
                    // An edge leaving the reduced set carries no meaningful rank any more.
                    // Resetting keeps ranks from drifting upward across days; correctness is
                    // safe because the edge is in the affected set, so its cell is revisited.
                    if (!keep) candidate.edge->meta.rank = 0;
                }
                if constexpr (collectTransferStats) {
                    if (keep) ++lc.minCandidatesKept;
                }
            }
        }
    }

private:
    Store& store_;
    const DynamicQueryData* queryData_{nullptr};
};

}  // namespace DynamicTB::Preprocessing
