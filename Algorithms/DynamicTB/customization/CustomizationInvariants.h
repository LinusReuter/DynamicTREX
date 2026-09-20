#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "../../../DataStructures/DynamicTimeTable/UpdateTypes.h"
#include "../../../Helpers/Types.h"
#include "CustomizationTypes.h"

namespace DynamicTB::Customization {

/**
 * @brief The three properties the incremental customization rests on and cannot observe.
 *
 * `IncrementalSeedPolicy` marks the cells one update disturbed and leaves every other cell's
 * ranks untouched. That is sound only if an *unmarked* cell's level pass would read exactly what
 * it read last minute. The kernel makes most of that easy: it never reads a time
 * (`CellTransferSearch` reads adjacency, ranks, the FIFO route structure and stop cells, and
 * `TripBased::EdgeLabel` carries no time at all), so a delay that leaves the minimized transfer
 * set alone cannot move a rank. What the argument still needs are three properties of code
 * *outside* the customization, none of which is visible from here:
 *
 *  1. **Flat routes mirror persistent routes** -- same order, one-to-one, only trip-less routes
 *     skipped. Were two persistent routes with equal stop sequences ever merged into one flat
 *     route and their trips ordered by time, a plain delay would reorder the merged FIFO group and
 *     change every search that touches it, with nothing in the `ChangeSummary` to seed from.
 *  2. **A trip's FIFO position changes only by extraction and re-insertion.** Reached-index
 *     pruning (`DynamicReachedIndex::update`) propagates along `firstTripOfRoute`, so a cell's
 *     found set depends on the route membership of the trips its edges point *into* -- trips that
 *     need not lie in that cell. The direct seed covers this only because a trip changes route
 *     exclusively through `extractTrip`, which lands it in `cancelledTrips`, whose cancellation
 *     phase clears the incoming edges and marks every source. A re-sort in place would bypass all
 *     of it, and a re-sort is exactly what a delay would provoke.
 *  3. **CSR rows are ordered by persistent target id.** The edge relaxation order inside one
 *     search is what makes it exact, so for an unmarked cell it has to be *stable across minutes*.
 *     It is, because the store sorts on the persistent target (`TransferStore::sort_outgoing`) and
 *     persistent ids never change: surviving targets shift position but never swap. A comparator
 *     on the flat target would be just as sorted and just as exact, and would silently break the
 *     incremental customization.
 *
 * All three are checked in full below rather than assumed, and a violation is a hard failure: an
 * under-marked seed set leaves ranks too low, which a query turns into wrong answers rather than
 * into slow ones. Debug builds prove them on every customization; release builds trust them.
 */
#ifdef NDEBUG
inline constexpr bool checkCustomizationInvariants = false;
#else
inline constexpr bool checkCustomizationInvariants = true;
#endif

/**
 * Invariant 1: every flat route is one persistent route, flat route ids ascend with persistent
 * route ids, each flat route's trips are its persistent route's trip list in the same order, and
 * the only persistent routes without a flat route are the ones with no trips.
 */
inline bool flatRoutesMirrorPersistentRoutes(const CellNetwork& net, std::string& error) {
    const auto& qd = net.qd->queryData;
    const auto& flatToPersistentRoute = net.qd->flatToPersistentRoute;
    const auto& persistentToFlatRoute = net.qd->persistentToFlatRoute;
    const auto& flatToPersistentTrip = net.qd->flatToPersistentTrip;
    const std::size_t flatRoutes = flatToPersistentRoute.size();
    std::ostringstream message;

    if (qd.firstTripOfRoute.size() != flatRoutes + 1) {
        message << "firstTripOfRoute has " << qd.firstTripOfRoute.size() << " entries for " << flatRoutes
                << " flat routes";
        error = message.str();
        return false;
    }

    for (std::size_t r = 0; r < flatRoutes; ++r) {
        const PersistentRouteId route = flatToPersistentRoute[r];
        if (static_cast<std::size_t>(route) >= net.data->routes().size()) {
            message << "flat route " << r << " maps to persistent route " << route << ", which does not exist";
            error = message.str();
            return false;
        }
        if (persistentToFlatRoute[static_cast<std::size_t>(route)] != RouteId(r)) {
            message << "flat route " << r << " does not round-trip through persistent route " << route;
            error = message.str();
            return false;
        }
        if (r > 0 && !(flatToPersistentRoute[r - 1] < route)) {
            message << "flat routes are not strictly ascending in persistent route id at flat route " << r
                    << " (persistent " << flatToPersistentRoute[r - 1] << " then " << route << ")";
            error = message.str();
            return false;
        }

        const auto& trips = net.data->getRoute(route).trips;
        const std::size_t begin = static_cast<std::size_t>(qd.firstTripOfRoute[r]);
        const std::size_t end = static_cast<std::size_t>(qd.firstTripOfRoute[r + 1]);
        if (end - begin != trips.size()) {
            message << "flat route " << r << " holds " << (end - begin) << " trips, persistent route " << route
                    << " holds " << trips.size();
            error = message.str();
            return false;
        }
        for (std::size_t k = 0; k < trips.size(); ++k) {
            if (flatToPersistentTrip[begin + k] != trips[k]) {
                message << "flat route " << r << " has trip " << flatToPersistentTrip[begin + k] << " at position "
                        << k << ", persistent route " << route << " has " << trips[k];
                error = message.str();
                return false;
            }
        }
    }

    const std::size_t persistentRoutes = net.data->routes().size();
    for (std::size_t r = 0; r < persistentRoutes; ++r) {
        if (net.data->getRoute(PersistentRouteId(r)).trips.empty()) continue;
        if (persistentToFlatRoute[r] == noRouteId) {
            message << "persistent route " << r << " has trips but no flat route";
            error = message.str();
            return false;
        }
    }
    return true;
}

/**
 * Invariant 3: within one CSR row the targets ascend in *persistent* event id, which is what
 * makes the relaxation order of an undisturbed source identical to last minute's.
 */
inline bool csrRowsAreSortedByPersistentTarget(const CellNetwork& net, std::string& error) {
    const auto& csr = *net.csr;
    const auto& flatToPersistentEvent = net.qd->flatToPersistentEvent;
    const std::size_t events = net.flatEventCount();

    for (std::size_t event = 0; event < events; ++event) {
        const std::size_t begin = static_cast<std::size_t>(csr.beginOut[event]);
        const std::size_t end = static_cast<std::size_t>(csr.beginOut[event + 1]);
        PersistentStopEventId previous(0);
        for (std::size_t edge = begin; edge < end; ++edge) {
            // `getStopEvent()` is the target event plus one -- `EdgeLabel::init`'s convention.
            const StopEventId flatTo = StopEventId(csr.labels[edge].getStopEvent() - 1);
            const PersistentStopEventId to = flatToPersistentEvent[static_cast<std::size_t>(flatTo)];
            if (edge > begin && !(previous < to)) {
                std::ostringstream message;
                message << "CSR row of flat event " << event << " is not ascending in persistent target id: "
                        << previous << " then " << to;
                error = message.str();
                return false;
            }
            previous = to;
        }
    }
    return true;
}

/**
 * @brief Invariant 2, which needs the previous minute's state: nothing re-orders a route's trips.
 *
 * Holds one `(route, position)` pair per persistent trip, snapshotted at the end of each seed
 * pass and compared at the start of the next one against the `ChangeSummary` of the update
 * applied in between -- the only licence for a difference. Two things are checked, both of them
 * about trips that update did *not* name:
 *
 *  - an unnamed trip may not change its persistent route, and
 *  - the unnamed trips of a route appear on it in the same relative order as before.
 *
 * Both exempt the trips in `cancelledTrips` and `addedTrips`, and that exemption is not a
 * loosening for convenience: a trip on either list had its incoming edges cleared and
 * re-discovered by the cancellation phase, which marks every source, so every cell whose pruning
 * could depend on where it sits has already been seeded. The exemption is also load-bearing rather
 * than theoretical -- `findCompatibleRoute` may hand a re-inserted trip back to the very route it
 * was extracted from, at a *different* FIFO index, so a delay really does re-order a route's trip
 * list. What must not happen is two *untouched* trips swapping, which is what this catches.
 *
 * Together those are exactly what makes reached-index pruning reproducible for a cell nobody
 * marked. Empty and inert in a release build.
 */
class TripFifoOrderWatch {
public:
    bool check(const CellNetwork& net, const DynamicTimeTable::ChangeSummary* changes, std::string& error) {
        if (previous_.empty()) return true;  // first customization of this run: no baseline yet
        std::ostringstream message;

        licensed_.clear();
        if (changes != nullptr) {
            licensed_.reserve(changes->cancelledTrips.size() + changes->addedTrips.size());
            for (const auto& cancelled : changes->cancelledTrips) licensed_.push_back(cancelled.tripId);
            for (const PersistentTripId trip : changes->addedTrips) licensed_.push_back(trip);
            std::sort(licensed_.begin(), licensed_.end());
        }
        const auto isLicensed = [this](const PersistentTripId trip) {
            return std::binary_search(licensed_.begin(), licensed_.end(), trip);
        };

        const auto& trips = net.data->trips();
        const std::size_t watched = std::min(previous_.size(), trips.size());
        for (std::size_t t = 0; t < watched; ++t) {
            const PersistentRouteId before = previous_[t].route;
            if (before == noPersistentRouteId) continue;  // was on no route: nothing to preserve
            const PersistentRouteId now = trips[t].route;
            if (now == before) continue;
            const PersistentTripId trip(t);
            if (isLicensed(trip)) continue;
            message << "trip " << trip << " moved from persistent route " << before << " to " << now
                    << " without appearing in cancelledTrips or addedTrips";
            error = message.str();
            return false;
        }

        const std::size_t routeCount = net.data->routes().size();
        for (std::size_t r = 0; r < routeCount; ++r) {
            const PersistentRouteId route(r);
            PersistentTripId lastTrip = noPersistentTripId;
            std::uint32_t lastPosition = 0;
            for (const PersistentTripId trip : net.data->getRoute(route).trips) {
                const std::size_t index = static_cast<std::size_t>(trip);
                if (index >= previous_.size() || previous_[index].route != route) continue;
                // A re-inserted trip may legitimately land at a new index of its old route.
                if (isLicensed(trip)) continue;
                if (lastTrip != noPersistentTripId && !(lastPosition < previous_[index].position)) {
                    message << "persistent route " << route << " re-ordered two untouched trips: " << trip
                            << " was at position " << previous_[index].position << ", now follows " << lastTrip
                            << " from position " << lastPosition;
                    error = message.str();
                    return false;
                }
                lastTrip = trip;
                lastPosition = previous_[index].position;
            }
        }
        return true;
    }

    void snapshot(const CellNetwork& net) {
        previous_.assign(net.data->trips().size(), Entry{});
        const std::size_t routeCount = net.data->routes().size();
        for (std::size_t r = 0; r < routeCount; ++r) {
            const auto& trips = net.data->getRoute(PersistentRouteId(r)).trips;
            for (std::size_t k = 0; k < trips.size(); ++k) {
                previous_[static_cast<std::size_t>(trips[k])] =
                    Entry{PersistentRouteId(r), static_cast<std::uint32_t>(k)};
            }
        }
    }

private:
    struct Entry {
        PersistentRouteId route = noPersistentRouteId;
        std::uint32_t position = 0;
    };

    std::vector<Entry> previous_;
    // Scratch for the licensed-trip set, reused across steps rather than rebuilt per call.
    std::vector<PersistentTripId> licensed_;
};

}  // namespace DynamicTB::Customization
