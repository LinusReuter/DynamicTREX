#pragma once

// Edge-by-edge comparison of two rank assignments. Used to check an incremental customization
// against a full one, and the dynamic customization against the static TREX builder.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../../Helpers/Types.h"
#include "../../TripBased/Query/Types.h"

namespace DynamicTB::Customization {

/**
 * @brief One edge whose rank differs between two customizations, in flat CSR space.
 */
struct RankMismatch {
    StopEventId from;
    StopEventId to;
    std::uint8_t leftRank;
    std::uint8_t rightRank;
};

/**
 * @brief The result of comparing two rank assignments over the same CSR.
 *
 * `underApproximated` is the count that matters for correctness: a rank *below* the reference is
 * an edge the query will prune when it should have kept it, i.e. a missed journey. A rank above
 * the reference only costs work. Keeping the two apart is what lets an inexact, raise-only result
 * be judged "drifted but sound" rather than simply "wrong".
 */
struct RankComparison {
    std::size_t edgesCompared = 0;
    std::size_t mismatches = 0;
    std::size_t underApproximated = 0;  // left < right: the dangerous direction
    std::size_t overApproximated = 0;   // left > right: safe, costs query work
    std::vector<RankMismatch> samples;

    bool identical() const noexcept { return mismatches == 0; }
};

/**
 * Copy the current ranks out of a CSR into a caller-owned buffer, so a later customization can be
 * diffed against them. Takes the buffer rather than returning one: a checked timeline snapshots
 * twice a minute, and at country scale that is a multi-MB allocation each time.
 */
inline void snapshotRanksInto(const TripBased::Transfers& csr, std::vector<std::uint8_t>& ranks) {
    ranks.resize(csr.labels.size());
    for (std::size_t edge = 0; edge < csr.labels.size(); ++edge) {
        ranks[edge] = csr.labels[edge].getRank();
    }
}

/// Convenience form for one-off callers (shell commands), where the allocation does not matter.
inline std::vector<std::uint8_t> snapshotRanks(const TripBased::Transfers& csr) {
    std::vector<std::uint8_t> ranks;
    snapshotRanksInto(csr, ranks);
    return ranks;
}

/**
 * Put a snapshot back. Computing the reference ranks overwrites the CSR, so the incremental result
 * -- which is the state a timeline continues from -- is restored from its snapshot afterwards
 * rather than recomputed.
 */
inline void restoreRanks(TripBased::Transfers& csr, const std::vector<std::uint8_t>& ranks) {
    const std::size_t edges = std::min(csr.labels.size(), ranks.size());
    for (std::size_t edge = 0; edge < edges; ++edge) {
        csr.labels[edge].setRank(ranks[edge]);
    }
}

/**
 * @brief Edge-by-edge rank diff of a snapshot against the CSR's current ranks.
 *
 * Positional: both sides must describe the same CSR, which is the case for "customize, snapshot,
 * customize again" on one export. For comparing across *different* exports (the incremental
 * store vs. a rebuild) the edges have to be matched by endpoints instead -- see
 * `compareRanksByEndpoint`.
 */
inline RankComparison compareRankSnapshot(const TripBased::Transfers& csr,
                                          const std::vector<std::uint8_t>& reference,
                                          const std::size_t maxSamples = 20) {
    RankComparison result;
    const std::size_t edges = std::min(csr.labels.size(), reference.size());
    result.edgesCompared = edges;
    for (std::size_t edge = 0; edge < edges; ++edge) {
        const std::uint8_t left = csr.labels[edge].getRank();
        const std::uint8_t right = reference[edge];
        if (left == right) continue;
        ++result.mismatches;
        if (left < right) {
            ++result.underApproximated;
        } else {
            ++result.overApproximated;
        }
        if (result.samples.size() < maxSamples) {
            // A positional scan does not know which CSR row an edge sits in, so the sample's
            // source is left unset; recovering it would cost a binary search over `beginOut` per
            // sample. `compareRanksByEndpoint` reports both endpoints.
            result.samples.push_back({noStopEvent, StopEventId(csr.labels[edge].getStopEvent() - 1), left, right});
        }
    }
    if (csr.labels.size() != reference.size()) {
        // Different edge counts mean the two sides are not the same graph at all; report it as a
        // mismatch rather than silently comparing a prefix.
        result.mismatches += std::max(csr.labels.size(), reference.size()) - edges;
    }
    return result;
}

/// A `(from, to, rank)` triple, used to compare rank assignments across two different exports.
struct RankedEdge {
    std::uint32_t from;
    std::uint32_t to;
    std::uint8_t rank;

    friend bool operator<(const RankedEdge& a, const RankedEdge& b) noexcept {
        return (a.from != b.from) ? a.from < b.from : a.to < b.to;
    }
};

/** Flatten a CSR into endpoint-keyed ranked edges, sorted, for cross-export comparison. */
inline std::vector<RankedEdge> extractRankedEdges(const TripBased::Transfers& csr) {
    std::vector<RankedEdge> edges;
    if (csr.beginOut.empty()) return edges;
    edges.reserve(csr.labels.size());
    for (std::size_t from = 0; from + 1 < csr.beginOut.size(); ++from) {
        for (Edge edge = csr.beginOut[from]; edge < csr.beginOut[from + 1]; ++edge) {
            edges.push_back({static_cast<std::uint32_t>(from),
                             static_cast<std::uint32_t>(csr.labels[edge].getStopEvent() - 1),
                             csr.labels[edge].getRank()});
        }
    }
    std::sort(edges.begin(), edges.end());
    return edges;
}

/**
 * @brief Compare two rank assignments that live in different exports, matching edges by
 * `(from, to)`.
 *
 * Reports an `edgesOnlyInLeft` / `edgesOnlyInRight` split as mismatches, because a rank
 * comparison over a differing edge set proves nothing -- check topology first.
 */
struct EndpointRankComparison {
    RankComparison ranks;
    std::size_t edgesOnlyInLeft = 0;
    std::size_t edgesOnlyInRight = 0;

    bool identical() const noexcept { return ranks.identical() && edgesOnlyInLeft == 0 && edgesOnlyInRight == 0; }
};

inline EndpointRankComparison compareRanksByEndpoint(const std::vector<RankedEdge>& left,
                                                     const std::vector<RankedEdge>& right,
                                                     const std::size_t maxSamples = 20) {
    EndpointRankComparison result;
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < left.size() && j < right.size()) {
        if (left[i] < right[j]) {
            ++result.edgesOnlyInLeft;
            ++i;
        } else if (right[j] < left[i]) {
            ++result.edgesOnlyInRight;
            ++j;
        } else {
            ++result.ranks.edgesCompared;
            if (left[i].rank != right[j].rank) {
                ++result.ranks.mismatches;
                if (left[i].rank < right[j].rank) {
                    ++result.ranks.underApproximated;
                } else {
                    ++result.ranks.overApproximated;
                }
                if (result.ranks.samples.size() < maxSamples) {
                    result.ranks.samples.push_back(
                        {StopEventId(left[i].from), StopEventId(left[i].to), left[i].rank, right[j].rank});
                }
            }
            ++i;
            ++j;
        }
    }
    result.edgesOnlyInLeft += left.size() - i;
    result.edgesOnlyInRight += right.size() - j;
    return result;
}

}  // namespace DynamicTB::Customization
