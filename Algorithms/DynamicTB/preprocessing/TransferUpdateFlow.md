# TransferUpdate – Data Flow

Below is the current function set and data flow for updates.

```
+--------------------------------------------------+
| buildInitialFullTransfers(queryData)             |
+--------------------------------------------------+
| 1) store.add_nodes(maxEventId)                   |
| 2) clear store                                   |
| 3) allowTemporaryInconsistent(true)              |
| 4) discover all outgoing transfers               |
|    (loop events -> updateOutgoingForEvent)       |
| 5) rebuild incoming / sync_barrier()             |
| 6) NO domination cleanup                         |
+--------------------------------------------------+

+--------------------------------------------------+
| applyFullUpdates(changes, queryData)             |
+--------------------------------------------------+
| 0) store.add_nodes(maxEventId)                   |
|                                                  |
| Phase 0: processRemovedRoutes                    |
|  - allowTemporaryInconsistent(true)              |
|  - clearRouteTransfers(route)                    |
|  - sync_barrier()                                |
|                                                  |
| Phase 1: processCancelledTrips                   |
|  - allowTemporaryInconsistent(true)              |
|  - redirectIncomingTransfers(trip)               |
|  - clearTripTransfers(trip) (outgoing+incoming)  |
|  - sync_barrier()                                |
|                                                  |
| Phase 2: Outgoing discovery                      |
|  - allowTemporaryInconsistent(true)              |
|  - added trips: loop events ->                   |
|      updateOutgoingForEvent                      |
|  - modified events: loop events ->               |
|      updateOutgoingForEvent                      |
|    (applies diffs via outgoing batches)          |
|  - diff preserves metadata for existing edges    |
|    (new edges get isMinimized=false)             |
|  - sync_barrier()                                |
|                                                  |
| Phase 3: Incoming discovery                      |
|  - allowTemporaryInconsistent(true)              |
|  - targets: added trips + modified events        |
|    (modified events include delayed arrivals;    |
|     do NOT expand tripsWithDelayedArrivals)      |
|  - loop target events -> updateIncomingForEvent  |
|     * compute desired sources:                   |
|        - connected source stops (footpaths)      |
|        - routes containing each source stop      |
|        - earliest feasible source events         |
|     * diff vs store.incoming(target)             |
|     * apply via incoming batch ops               |
|       (new edges get isMinimized=false)          |
|     * on insert: domination cleanup on target    |
|       route (remove later dominated targets)     |
|  - sync_barrier()                                |
|                                                  |
| Domination cleanup (incoming-only)               |
|  - triggered INSIDE updateIncomingForEvent       |
|  - if a removed edge was isMinimized=true,       |
|    mark source trip for re-minimization          |
+--------------------------------------------------+

+--------------------------------------------------+
| buildInitialMinimizedTransfers(queryData)        |
+--------------------------------------------------+
| loop trips: clearMinimizationFlags +             |
|              recomputeMinimizedForTrip           |
+--------------------------------------------------+

+--------------------------------------------------+
| updateMinimizedTransfers(trips, queryData)       |
+--------------------------------------------------+
| loop trips: clearMinimizationFlags +             |
|              recomputeMinimizedForTrip           |
+--------------------------------------------------+

Candidate selection guidance:
- Any source trip whose outgoing diff changed (add/remove).
- Any source trip that gained/removed incoming transfers due to redirection.
- Any source trip whose existing transfers were removed by domination cleanup,
  but only if a removed edge had isMinimized=true.
- Any source trip that transfers into a trip with delayed arrivals.
```
