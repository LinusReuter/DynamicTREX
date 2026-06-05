# TransferUpdate – Data Flow

+--------------------------------------------------+
| buildInitialFullTransfers(queryData)             |
+--------------------------------------------------+
| 1) store.add_nodes(maxEventId)                   |
| 2) clear store                                   |
| 3) allowTemporaryInconsistent(true)              |
| 4) discover all outgoing transfers               |
|    (loop events -> updateOutgoingForEvent)       |
| 5) rebuild incoming / sync_barrier()             |
+--------------------------------------------------+

+--------------------------------------------------+
| applyFullUpdates -> ret {MinimizationCandidates} |
+--------------------------------------------------+
| 0) store.add_nodes(maxEventId)                   |
|                                                  |
| Phase 0/1: processCancelledTrips                 |
|  - allowTemporaryInconsistent(true)              |
|  - if nextActiveTrip exists: redirect incoming   |
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
|    (NOTE: `tripsWithDelayedArrivals` is for      |
|     minimization only and does not trigger this) |
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
|                                                  |
| During all phases, the implementation collects a |
| set of trips that require re-minimization based  |
| on the following rules:                          |
|                                                  |
| - Source trip of any transfer that is added or   |
|   removed during a diff/apply step.              |
| - Source trip of any incoming transfer that is   |
|   redirected during trip cancellation.           |
| - Source trip of any transfer removed by         |
|   domination cleanup if it had `isMinimized=true`|
| - Source trip of any transfer pointing to a trip |
|   in the `tripsWithDelayedArrivals` list.        |
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
