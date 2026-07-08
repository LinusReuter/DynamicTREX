#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

/// Interface for DAG TransferStore.
/// - Node IDs are sequential: 0..n-1
/// - No multi-edges: (from, to) is unique
/// - Adjacency views are sorted by neighbor ID and contain unique entries
/// - Incoming/outgoing may be temporarily unsynchronized until `sync_barrier()`
/// Usage
/// - **Init (full rebuild):** `begin_outgoing_init(maxNodeId)` → clear outgoing → `add_outgoing_edges_init(...)` →
/// `finish_outgoing_init()`
/// - **Phase 0 Removed lines:** `allowTemporaryInconsistent(true)` → apply clears → `sync_barrier()`
/// - **Phase 1 Trip cancellation:** same pattern; redirections only add new edges not touching any cancelled Event,
/// async allowed
/// - **Barrier**
/// - **Phase 2 Outgoing:** update outgoing with temporary inconsistency enabled → `sync_barrier()`
/// - **Phase 3 Incoming:** update incoming with temporary inconsistency enabled → `sync_barrier()`
/// - **Minimization**

template <typename NodeID, typename EdgeMeta>
class ITransferStore {
public:
    /// Outgoing edge record stored per source node.
    struct OutEdge {
        NodeID to;
        EdgeMeta meta;
    };

    using node_id_type = NodeID;
    using edge_meta_type = EdgeMeta;
    using outgoing_span = std::span<const OutEdge>;
    using mutable_outgoing_span = std::span<OutEdge>;
    using incoming_span = std::span<const NodeID>;
    using batch_id_type = std::size_t;

    enum class Direction : std::uint8_t { Outgoing, Incoming };

    virtual ~ITransferStore() = default;

    // === Topology ===

    /// Current number of nodes (valid IDs are 0..node_count()-1).
    virtual std::size_t node_count() const noexcept = 0;

    /// Append nodes to reach maxNodeId
    virtual void add_nodes(NodeID maxNodeId) = 0;

    // ===  Adjacency views ===

    /// Outgoing adjacency view for `from` (read-only).
    virtual outgoing_span outgoing_unsorted(NodeID from) = 0;

    /// Incoming adjacency view for `to` (read-only sources only).
    virtual incoming_span incoming_unsorted(NodeID to) = 0;

    /// Atomically copy the outgoing adjacency of `from` into `out` (cleared first).
    /// Unlike outgoing_unsorted()/outgoing_sorted(), the returned data is a stable
    /// snapshot: it is safe to iterate while other threads perform concurrent
    /// opposite-direction (incoming) maintenance that may reallocate the underlying
    /// storage. Prefer this over holding a span across concurrent writes.
    /// Default implementation copies the unsorted view (adequate for stores that do
    /// not reallocate under concurrency); thread-safe stores must lock internally.
    virtual void copy_outgoing(NodeID from, std::vector<OutEdge>& out) {
        const auto span = outgoing_unsorted(from);
        out.assign(span.begin(), span.end());
    }

    /// Atomically copy the incoming adjacency (source node IDs) of `to` into `out`
    /// (cleared first). Same stable-snapshot guarantee as copy_outgoing(): safe to
    /// iterate while other threads perform concurrent opposite-direction (outgoing)
    /// maintenance that may reallocate the underlying storage.
    /// Default implementation copies the unsorted view; thread-safe stores must lock
    /// internally.
    virtual void copy_incoming(NodeID to, std::vector<NodeID>& out) {
        const auto span = incoming_unsorted(to);
        out.assign(span.begin(), span.end());
    }

    // === Sorted adjacency views ===

    /// Outgoing adjacency view for `from` (read-only).
    /// Contract: sorted by `to`, unique, and stable until commit_batch() for
    /// the same node + direction (Outgoing). Opposite-direction maintenance
    /// may update other nodes concurrently (no cross-direction stability guarantee).
    virtual outgoing_span outgoing_sorted(NodeID from) = 0;

    /// Mutable outgoing adjacency view for `from`, in arbitrary (storage) order and
    /// WITHOUT sorting the underlying list. Intended for phases that own `from`
    /// exclusively (e.g. minimization, where each source stop event belongs to a
    /// single trip) and only mutate edge metadata in place. The span is valid until
    /// the next structural mutation of `from`.
    virtual mutable_outgoing_span outgoing_mutable(NodeID from) = 0;

    /// Incoming adjacency view for `to` (read-only sources only).
    /// Contract: sorted, unique, and stable until commit_batch() for
    /// the same node + direction (Incoming). Opposite-direction maintenance
    /// may update other nodes concurrently (no cross-direction stability guarantee).
    virtual incoming_span incoming_sorted(NodeID to) = 0;

    // === Edge operations (non-batch) ===

    /// Returns read-only EdgeMeta of an given Edge.
    virtual EdgeMeta readEdgeMeta(NodeID from, NodeID to) const = 0;

    /// Add edge (from -> to) with metadata. Returns false if edge already exists.
    virtual bool add_edge(NodeID from, NodeID to, const EdgeMeta& meta) = 0;

    /// Remove edge (from -> to). Returns false if edge does not exist.
    virtual bool remove_edge(NodeID from, NodeID to) = 0;

    /// Replace metadata for edge (from -> to). Returns false if edge missing.
    virtual bool update_edge_meta(NodeID from, NodeID to, const EdgeMeta& meta) = 0;

    // === Clears ===

    /// Remove all outgoing edges for `from`.
    virtual void clear_outgoing(NodeID from) = 0;

    /// Remove all incoming edges for `to`.
    virtual void clear_incoming(NodeID to) = 0;

    /// Remove all incoming edges for `to`, appending each removed edge as
    /// (source, meta) to `removed`. The meta is read from the source's outgoing
    /// record while it is erased, so thread-safe stores return it without an extra
    /// scan and without racing concurrent opposite-direction maintenance.
    /// Default: copy_incoming() + readEdgeMeta() + clear_incoming() (fine for stores
    /// that maintain full consistency and do not need internal locking).
    virtual void clear_incoming_with_meta(NodeID to, std::vector<std::pair<NodeID, EdgeMeta>>& removed) {
        std::vector<NodeID> sources;
        copy_incoming(to, sources);
        removed.reserve(removed.size() + sources.size());
        for (const NodeID from : sources) {
            removed.emplace_back(from, readEdgeMeta(from, to));
        }
        clear_incoming(to);
    }

    /// Clears the complete store
    virtual void clear() = 0;

    // === Phase / consistency ===

    /// Allow or disallow temporary inconsistency.
    /// When false (default), all methods return fully synchronized state.
    /// When true, internal async work may be deferred until `sync_barrier()`.
    virtual void allowTemporaryInconsistent(bool allow) = 0;

    /// Rebuild incoming adjacency from outgoing (e.g., after init).
    virtual void rebuild_incoming() = 0;

    /// Block until all queued sync work has been applied.
    virtual void sync_barrier() = 0;

    // === Initial build helpers (outgoing-only, incoming built in bulk) ===

    /// Begin initial build mode (outgoing only). Implementations may defer any
    /// incoming maintenance until `finish_outgoing_init()`.
    /// Default: add_nodes(maxNodeId) and allowTemporaryInconsistent(true).
    virtual void begin_outgoing_init(NodeID maxNodeId) {
        add_nodes(maxNodeId);
        allowTemporaryInconsistent(true);
    }

    /// Add a single outgoing edge during initial build.
    /// Default: forwards to add_edge().
    virtual void add_outgoing_edge_init(NodeID from, NodeID to, const EdgeMeta& meta) {
        add_edge(from, to, meta);
    }

    /// Add multiple outgoing edges during initial build (same meta for all).
    /// Default: forwards to add_outgoing_edge_init() in a loop.
    virtual void add_outgoing_edges_init(NodeID from, std::span<const NodeID> to,
                                         const EdgeMeta& meta) {
        for (const NodeID dst : to) {
            add_outgoing_edge_init(from, dst, meta);
        }
    }

    /// Finish initial build: rebuild incoming and synchronize.
    /// Default: rebuild_incoming() -> sync_barrier() -> allowTemporaryInconsistent(false).
    virtual void finish_outgoing_init() {
        rebuild_incoming();
        sync_barrier();
        allowTemporaryInconsistent(false);
    }

    // === Batched updates (single node, single direction) ===

    /// Begin a batch for a single node. The batch ID must be used for all
    /// operations in that batch. The batch must only touch one node.
    /// Direction::Incoming: batch applies to incoming adjacency of `node`.
    /// Direction::Outgoing: batch applies to outgoing adjacency of `node`.
    virtual batch_id_type begin_batch(NodeID node, Direction dir) = 0;

    /// Commit a previously started batch.
    virtual void commit_batch(batch_id_type batch) = 0;

    /// Batch-scoped outgoing edge operations (batch node is the source).
    virtual bool add_outgoing_edge(batch_id_type batch, NodeID to,
                                   const EdgeMeta& meta) = 0;
    virtual bool remove_outgoing_edge(batch_id_type batch, NodeID to) = 0;

    /// Batch-scoped incoming edge operations (batch node is the target).
    virtual bool add_incoming_edge(batch_id_type batch, NodeID from, const EdgeMeta& meta) = 0;
    virtual bool remove_incoming_edge(batch_id_type batch, NodeID from) = 0;

    // === Optional performance hooks (no-ops by default) ===
    virtual void reserve_outgoing(NodeID, std::size_t) {}
    virtual void reserve_incoming(NodeID, std::size_t) {}

    virtual int out_degree(NodeID node_id) = 0;

    virtual std::unordered_map<u_int64_t, u_int64_t> edgeDegreeDistrebutionOut() const noexcept = 0;
    virtual std::unordered_map<u_int64_t, u_int64_t> edgeDegreeDistrebutionIn() const noexcept = 0;
};