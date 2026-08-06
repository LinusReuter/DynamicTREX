# TransferUpdate – Components & Data Flow

## Components

| Header | Responsibility |
|---|---|
| `TransferTypes.h` | `TransferMeta` (`isMinimized`, `rank`), `StopLabel`, `MinTarget`, `PendingDominationCleanup`, `RankRaise`, `sortUnique`, the `collectTransferStats` gate |
| `AffectedEventSink.h` | `NullAffectedSink` / `AffectedEventCollector` / `AffectedEvents` — the TREX level-0 affected-event delta |
| `TransferDiscovery.h` | Pure timetable queries: which transfers exist. Never touches the store. Owns `DiscoveryWorkspace` (per-thread scratch) |
| `TransferStoreMutator.h` | The only store-mutating layer: sorted diffs, batch apply, domination cleanup |
| `TransferMinimizer.h` | The reduced-set kernel: `reduceTransfersForTrip` + parallel driver. Owns `MinimizationWorkspace` |
| `TransferExport.h` | Persistent store → flat CSR (`TripBased::Transfers`), carrying `rank`; sparse `applyRankRaises` write-back |
| `TransferUpdate.h` | The phase driver that composes the above |

`TransferUpdate<Store, AffectedSink>` is templated on the **concrete** store (so per-edge
calls inline rather than dispatching through `ITransferStore`) and on the affected-set sink
(so with `NullAffectedSink` the whole TREX delta collection compiles away).

## buildInitialFullTransfers(queryData)

```
1) store.clear()
2) store.begin_outgoing_init(maxEventId)
3) parallel over persistent events:
     TransferDiscovery::computeOutgoingTransfers -> store.add_outgoing_edges_init
4) store.finish_outgoing_init()   // rebuilds incoming in bulk
```

## applyFullUpdates(changes, queryData) → re-minimization targets → minimization

```
0) store.add_nodes(maxEventId); allowTemporaryInconsistent(true)

Phase 1: Cancellations                                      [parallel, then sync_barrier]
  - clear_outgoing(event)
  - clear_incoming_with_meta(event) -> per removed edge with isMinimized:
        flag source trip for re-minimization
        AffectedSink::markEvent(source)          <- reduced set shrank at the source

collectDiscoveryTargets(changes) -> outgoing / incoming worklists

Phase 2: Outgoing discovery                                 [parallel, then sync_barrier]
  - computeOutgoingTransfers -> applyOutgoingDiff (mergeSortedDiff vs. stored outgoing)
  - flag the trip iff an edge was added, or a MINIMIZED edge removed
  - removed minimized edge -> AffectedSink::markEdgeChanged(from, to)
  - new edges enter with TransferMeta{} (not minimized, rank 0)

Phase 3: Incoming discovery                                 [parallel, then sync_barrier]
  - computeIncomingTransfers -> applyIncomingDiff
  - every add/remove flags the source trip
  - removals mark the affected set unconditionally: incoming storage carries no metadata,
    so we cannot tell whether the mirrored outgoing edge was minimized. Over-approximating
    is safe; missing an entry would leave a rank too low.
  - inserts record a deferred PendingDominationCleanup

Domination cleanup                                          [sequential]
  - removes an outgoing edge dominated by a newly inserted one
  - if it was minimized: flag the source trip + mark the affected set

Phase 4: Changed arrivals                                   [parallel]
  (a) SELF     : re-minimize the changed trip from maxChangedIndex down
  (b) UPSTREAM : re-minimize sources feeding stops at index <= maxChangedIndex

aggregateMaxByTrip(targets)      // one entry per trip, max warm-start boundary
updateMinimizedTransfers(targets)
```

## Minimization (`TransferMinimizer::reduceTransfersForTrip`)

Scans a trip's stops high→low carrying a per-stop min-arrival profile.
Stops above the warm-start boundary only **replay** already-kept edges (no decisions, so no
affected-set entries). At and below it, candidates are sorted by destination arrival and
folded; every `isMinimized` **flip** is reported to the sink, and an edge leaving the
reduced set has its `rank` reset to 0.


