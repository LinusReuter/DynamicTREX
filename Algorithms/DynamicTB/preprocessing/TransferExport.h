#pragma once

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../Helpers/MultiThreading.h"
#include "../../../Helpers/Types.h"
#include "../../DynamicTimeTable/BuildQueryData.h"
#include "../../TripBased/Query/Types.h"
#include "TransferTypes.h"

namespace DynamicTB::Preprocessing {

/**
 * @brief Exports the persistent transfer store into the flat CSR the query engine consumes.
 *
 * `TransferMeta::rank` rides along for free with the existing three-pass export: it is written
 * into the 5 rank bits `TripBased::EdgeLabel` already has, so the exported CSR is a ranked (TREX)
 * transfer graph without any extra pass.
 *
 * The reverse direction (customization results -> persistent ranks) is deliberately sparse:
 * see `applyRankRaises`.
 */
template <class Store>
class TransferExporter {
public:
    using DynamicQueryData = DynamicTimeTable::Algo::DynamicQueryData;
    using NodeID = PersistentStopEventId;

    explicit TransferExporter(Store& store) noexcept : store_(store) {}

    [[nodiscard]] TripBased::Transfers exportFull(const DynamicQueryData& queryData, const int numberOfThreads) const {
        return exportImpl<false>(queryData, numberOfThreads);
    }

    [[nodiscard]] TripBased::Transfers exportReduced(const DynamicQueryData& queryData,
                                                     const int numberOfThreads) const {
        return exportImpl<true>(queryData, numberOfThreads);
    }

    /**
     * @brief Write customization results back into the persistent store.
     *
     * Sparse by construction: the customization knows exactly which edges it raised, so this
     * costs O(raises), not O(reduced edges). A full write-back every minute would be tens of
     * millions of random store lookups at country scale.
     */
    void applyRankRaises(std::span<const RankRaise> raises, const int numberOfThreads = 1) const {
        const int threads = std::max(1, numberOfThreads);
        omp_set_num_threads(threads);
#pragma omp parallel for schedule(dynamic, 1024) if (threads > 1)
        for (std::size_t i = 0; i < raises.size(); ++i) {
            const RankRaise& raise = raises[i];
            for (auto& edge : store_.outgoing_mutable(raise.from)) {
                if (edge.to != raise.to) continue;
                if (edge.meta.rank < raise.rank) edge.meta.rank = raise.rank;
                break;
            }
        }
    }

private:
    /**
     * @brief Unified multi-pass implementation for exporting graph structures.
     */
    template <bool OnlyMinimized>
    [[nodiscard]] TripBased::Transfers exportImpl(const DynamicQueryData& queryData, const int numberOfThreads) const {
        const auto& qd = queryData.queryData;
        const std::size_t flatEventCount = qd.eventLookup.size();
        const std::size_t persistentCount = queryData.persistentToFlatEvent.size();

        assertEdgeLabelCapacity(flatEventCount, qd.routeOfTrip.size());

        std::vector<Edge> beginOut(flatEventCount + 1, Edge(0));
        const int threads = std::max(1, numberOfThreads);
        omp_set_num_threads(threads);

// Pass 1: Directly map out-degrees to the flat event indices in parallel
#pragma omp parallel if (threads > 1)
        {
            if (omp_get_num_threads() > 1) pinThreadToCoreId(omp_get_thread_num() % numberOfCores());
#pragma omp for schedule(dynamic, 1024)
            for (std::size_t pFrom = 0; pFrom < persistentCount; ++pFrom) {
                const StopEventId flatFrom = queryData.persistentToFlatEvent[pFrom];
                if (flatFrom == noStopEvent) continue;

                Edge degree = Edge(0);
                if constexpr (OnlyMinimized) {
                    for (const auto& [to, meta] : store_.outgoing_sorted(NodeID(pFrom))) {
                        if (meta.isMinimized && queryData.persistentToFlatEvent[to] != noStopEvent) {
                            ++degree;
                        }
                    }
                } else {
                    degree = Edge(store_.out_degree(NodeID(pFrom)));
                }
                beginOut[static_cast<std::size_t>(flatFrom) + 1] = degree;
            }
        }

        // Pass 2: Prefix sum
        for (std::size_t i = 1; i < beginOut.size(); ++i) {
            beginOut[i] = Edge(beginOut[i] + beginOut[i - 1]);
        }

        const std::size_t edgeCount = beginOut.back();
        std::vector<TripBased::EdgeLabel> labels(edgeCount);
        std::vector<int> travelTime(edgeCount);

#pragma omp parallel if (threads > 1)
        {
            if (omp_get_num_threads() > 1) pinThreadToCoreId(omp_get_thread_num() % numberOfCores());
#pragma omp for schedule(dynamic, 1024)
            for (std::size_t pFrom = 0; pFrom < persistentCount; ++pFrom) {
                const StopEventId flatFrom = queryData.persistentToFlatEvent[pFrom];
                if (flatFrom == noStopEvent) continue;

                const Time fromArrivalTime = queryData.arrivalTimeOfEvent(flatFrom);
                Edge currentEdgeOffset = beginOut[flatFrom];

                for (const auto& [to, meta] : store_.outgoing_sorted(NodeID(pFrom))) {
                    if constexpr (OnlyMinimized) {
                        if (!meta.isMinimized) continue;
                    }
                    const StopEventId flatTo = queryData.persistentToFlatEvent[to];
                    if (flatTo == noStopEvent) continue;

                    const Edge exportEdge = currentEdgeOffset++;
                    const TripId trip = queryData.tripOfEvent(flatTo);
                    labels[exportEdge].init(flatTo, trip, qd.firstStopEventOfTrip[trip]);
                    // The TREX rank lives in EdgeLabel's spare bits, so the exported CSR is
                    // already the ranked graph the query prunes with.
                    labels[exportEdge].setRank(meta.rank);
                    travelTime[exportEdge] =
                        static_cast<int>(queryData.departureTimeOfEvent(flatTo) - fromArrivalTime);
                }
            }
        }
        return {std::move(beginOut), std::move(labels), std::move(travelTime)};
    }

    /**
     * @brief TripBased::EdgeLabel packs the target trip into 23 bits and the trip's first
     * stop event into 27 bits. Country-scale instances sit close enough to the event limit
     * that a silent truncation is a real risk -- fail loudly instead.
     */
    static void assertEdgeLabelCapacity(const std::size_t flatEventCount, const std::size_t tripCount) {
        constexpr std::size_t maxEvents = std::size_t(1) << 27;
        constexpr std::size_t maxTrips = std::size_t(1) << 23;
        if (flatEventCount > maxEvents) {
            throw std::runtime_error("TransferExporter: " + std::to_string(flatEventCount) +
                                     " stop events exceed the 27-bit EdgeLabel field (max " +
                                     std::to_string(maxEvents) + "); widen TripBased::EdgeLabel.");
        }
        if (tripCount > maxTrips) {
            throw std::runtime_error("TransferExporter: " + std::to_string(tripCount) +
                                     " trips exceed the 23-bit EdgeLabel field (max " + std::to_string(maxTrips) +
                                     "); widen TripBased::EdgeLabel.");
        }
    }

    Store& store_;
};

}  // namespace DynamicTB::Preprocessing
