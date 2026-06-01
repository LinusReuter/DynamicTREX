#pragma once

#include <cstddef>
#include <span>

/// Interface for DAG TransferStore.
/// - Node IDs are sequential: 0..n-1
/// - No multi‑edges: (from, to) is unique
/// - Incoming/outgoing may be temporarily unsynchronized until `sync_barrier()`
/// Usage
/// - **Init (full rebuild):** `begin_outgoing_init(maxNodeId)` → clear outgoing → `add_outgoing_edges_init(...)` → `finish_outgoing_init()`
/// - **Phase 0 Removed lines:** `allowTemporaryInconsistent(true)` → apply clears → `sync_barrier()`
/// - **Phase 1 Trip cancellation:** same pattern; redirections only add new edges not touching any cancelled Event, async allowed
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
    using incoming_span = std::span<const NodeID>;
    using batch_id_type = std::size_t;

    virtual ~ITransferStore() = default;

    /// Current number of nodes (valid IDs are 0..node_count()-1).
    virtual std::size_t node_count() const noexcept = 0;

    /// Append nodes to reach maxNodeId
    virtual void add_nodes(NodeID maxNodeId) = 0;

    /// Outgoing adjacency view for `from` (read‑only).
    virtual outgoing_span outgoing(NodeID from) const = 0;

    /// Incoming adjacency view for `to` (read‑only sources only).
    virtual incoming_span incoming(NodeID to) const = 0;

    /// Returns read-only EdgeMeta of an given Edge.
    virtual EdgeMeta readEdgeMeta(NodeID from, NodeID to) const = 0;

    /// Add edge (from -> to) with metadata. Returns false if edge already exists.
    virtual bool add_edge(NodeID from, NodeID to, const EdgeMeta& meta) = 0;

    /// Remove edge (from -> to). Returns false if edge does not exist.
    virtual bool remove_edge(NodeID from, NodeID to) = 0;

    /// Replace metadata for edge (from -> to). Returns false if edge missing.
    virtual bool update_edge_meta(NodeID from, NodeID to, const EdgeMeta& meta) = 0;

    /// Remove all outgoing edges for `from`.
    virtual void clear_outgoing(NodeID from) = 0;

    /// Remove all incoming edges for `to`.
    virtual void clear_incoming(NodeID to) = 0;

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

    /// Clears the complete store
    virtual void clear() = 0;

    /// Begin a batch for a single node. The batch ID must be used for all
    /// operations in that batch. The batch must only touch one node.
    /// incoming=true: batch applies to incoming adjacency of `node`.
    /// incoming=false: batch applies to outgoing adjacency of `node`.
    virtual batch_id_type begin_batch(NodeID node, bool incoming) = 0;

    /// Commit a previously started batch.
    virtual void commit_batch(batch_id_type batch) = 0;

    /// Batch‑scoped outgoing edge operations (batch node is the source).
    virtual bool add_outgoing_edge(batch_id_type batch, NodeID to,
                                   const EdgeMeta& meta) = 0;
    virtual bool remove_outgoing_edge(batch_id_type batch, NodeID to) = 0;

    /// Batch‑scoped incoming edge operations (batch node is the target).
    virtual bool add_incoming_edge(batch_id_type batch, NodeID from, const EdgeMeta& meta) = 0;
    virtual bool remove_incoming_edge(batch_id_type batch, NodeID from) = 0;

    /// Optional performance hooks (no‑ops by default).
    virtual void reserve_outgoing(NodeID, std::size_t) {}
    virtual void reserve_incoming(NodeID, std::size_t) {}
};
