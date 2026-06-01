#pragma once

#include <cstddef>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <vector>

#include "ITransferStore.h"
#include "../Graph/Classes/DynamicGraph.h"
#include "../../Helpers/Types.h"

/// Adapter that maps ITransferStore to an existing DynamicGraph.
///
/// Notes:
/// - Thread-safe via a global shared_mutex.
/// - Always maintains consistent incoming/outgoing (DynamicGraph invariant).
/// - Batching is ignored, but batch-scoped operations forward using a
///   per-thread active node (set by begin_batch).
/// - Temporary inconsistency is ignored (no-op).
/// - Outgoing/incoming spans are backed by thread-local snapshots.

template <typename Graph, typename NodeID, typename EdgeMeta, AttributeNameType META_ATTR>
class DynamicGraphTransferStore : public ITransferStore<NodeID, EdgeMeta> {
public:
    using Base = ITransferStore<NodeID, EdgeMeta>;
    using OutEdge = typename Base::OutEdge;
    using outgoing_span = typename Base::outgoing_span;
    using incoming_span = typename Base::incoming_span;
    using batch_id_type = typename Base::batch_id_type;

    static_assert(Graph::template HasEdgeAttribute<META_ATTR>(AttributeNameWrapper<META_ATTR>()),
                  "DynamicGraph is missing required EdgeMeta attribute");
    static_assert(std::is_same_v<typename Graph::template EdgeAttributeType<META_ATTR>, EdgeMeta>,
                  "EdgeMeta type does not match DynamicGraph edge attribute type");

    explicit DynamicGraphTransferStore(Graph& graph) : graph_(graph) {}

    std::size_t node_count() const noexcept override {
        std::shared_lock lock(mutex_);
        return graph_.numVertices();
    }

    void add_nodes(NodeID maxNodeId) override {
        std::unique_lock lock(mutex_);
        const std::size_t target = to_index(maxNodeId) + 1;
        const std::size_t current = graph_.numVertices();
        if (target > current) {
            graph_.addVertices(target - current);
        }
    }

    outgoing_span outgoing(NodeID from) const override {
        std::shared_lock lock(mutex_);
        const Vertex v = to_vertex(from);
        if (!graph_.isVertex(v)) {
            return {};
        }
        static thread_local std::vector<OutEdge> tls_out;
        tls_out.clear();
        tls_out.reserve(graph_.outDegree(v));
        for (const Edge edge : graph_.edgesFrom(v)) {
            if (!graph_.isEdge(edge)) {
                continue;
            }
            const Vertex to = graph_.get(ToVertex, edge);
            const EdgeMeta meta = graph_.get(meta_attr_, edge);
            tls_out.push_back(OutEdge{to_node(to), meta});
        }
        return outgoing_span(tls_out.data(), tls_out.size());
    }

    incoming_span incoming(NodeID to) const override {
        std::shared_lock lock(mutex_);
        const Vertex v = to_vertex(to);
        if (!graph_.isVertex(v)) {
            return {};
        }
        static thread_local std::vector<NodeID> tls_in;
        tls_in.clear();
        const auto& edges = graph_.edgesTo(v);
        tls_in.reserve(edges.size());
        for (const Edge edge : edges) {
            if (!graph_.isEdge(edge)) {
                continue;
            }
            const Vertex from = graph_.get(FromVertex, edge);
            tls_in.push_back(to_node(from));
        }
        return incoming_span(tls_in.data(), tls_in.size());
    }

    EdgeMeta readEdgeMeta(NodeID from, NodeID to) const override {
        std::shared_lock lock(mutex_);
        const Vertex v_from = to_vertex(from);
        const Vertex v_to = to_vertex(to);
        if (!graph_.isVertex(v_from) || !graph_.isVertex(v_to)) {
            return EdgeMeta{};
        }
        const Edge edge = graph_.findEdge(v_from, v_to);
        if (!graph_.isEdge(edge)) {
            return EdgeMeta{};
        }
        return graph_.get(meta_attr_, edge);
    }

    bool add_edge(NodeID from, NodeID to, const EdgeMeta& meta) override {
        std::unique_lock lock(mutex_);
        const Vertex v_from = to_vertex(from);
        const Vertex v_to = to_vertex(to);
        if (!graph_.isVertex(v_from) || !graph_.isVertex(v_to)) {
            return false;
        }
        const Edge existing = graph_.findEdge(v_from, v_to);
        if (graph_.isEdge(existing)) {
            return false;
        }
        Edge edge = graph_.addEdge(v_from, v_to);
        graph_.set(meta_attr_, edge, meta);
        return true;
    }

    bool remove_edge(NodeID from, NodeID to) override {
        std::unique_lock lock(mutex_);
        const Vertex v_from = to_vertex(from);
        const Vertex v_to = to_vertex(to);
        if (!graph_.isVertex(v_from) || !graph_.isVertex(v_to)) {
            return false;
        }
        const Edge edge = graph_.findEdge(v_from, v_to);
        if (!graph_.isEdge(edge)) {
            return false;
        }
        graph_.deleteEdge(edge);
        return true;
    }

    bool update_edge_meta(NodeID from, NodeID to, const EdgeMeta& meta) override {
        std::unique_lock lock(mutex_);
        const Vertex v_from = to_vertex(from);
        const Vertex v_to = to_vertex(to);
        if (!graph_.isVertex(v_from) || !graph_.isVertex(v_to)) {
            return false;
        }
        const Edge edge = graph_.findEdge(v_from, v_to);
        if (!graph_.isEdge(edge)) {
            return false;
        }
        graph_.set(meta_attr_, edge, meta);
        return true;
    }

    void clear_outgoing(NodeID from) override {
        std::unique_lock lock(mutex_);
        const Vertex v_from = to_vertex(from);
        if (!graph_.isVertex(v_from)) {
            return;
        }
        graph_.deleteAllOutgoingEdges(v_from);
    }

    void clear_incoming(NodeID to) override {
        std::unique_lock lock(mutex_);
        const Vertex v_to = to_vertex(to);
        if (!graph_.isVertex(v_to)) {
            return;
        }
        graph_.deleteAllIncomingEdges(v_to);
    }

    void clear() override {
        std::unique_lock lock(mutex_);
        graph_.clear();
    }

    void allowTemporaryInconsistent(bool) override {
        // No-op: DynamicGraph always keeps incoming/outgoing consistent.
    }

    void rebuild_incoming() override {
        // No-op: DynamicGraph maintains incoming edges as an invariant.
    }

    void sync_barrier() override {
        // No-op: no async sync work.
    }

    batch_id_type begin_batch(NodeID node, bool incoming) override {
        auto& ctx = batch_context();
        ctx.node = node;
        ctx.incoming = incoming;
        return batch_id_type{0};
    }

    void commit_batch(batch_id_type) override {
    }

    bool add_outgoing_edge(batch_id_type, NodeID to, const EdgeMeta& meta) override {
        auto& ctx = batch_context();
        if (ctx.incoming) {
            return false;
        }
        return add_edge(ctx.node, to, meta);
    }

    bool remove_outgoing_edge(batch_id_type, NodeID to) override {
        auto& ctx = batch_context();
        if (ctx.incoming) {
            return false;
        }
        return remove_edge(ctx.node, to);
    }

    bool add_incoming_edge(batch_id_type, NodeID from, const EdgeMeta& meta) override {
        auto& ctx = batch_context();
        if (!ctx.incoming) {
            return false;
        }
        return add_edge(from, ctx.node, meta);
    }

    bool remove_incoming_edge(batch_id_type, NodeID from) override {
        auto& ctx = batch_context();
        if (!ctx.incoming) {
            return false;
        }
        return remove_edge(from, ctx.node);
    }

private:
    struct BatchContext {
        NodeID node{};
        bool incoming{false};
    };

    static BatchContext& batch_context() {
        thread_local BatchContext ctx{};
        return ctx;
    }

    static std::size_t to_index(NodeID id) {
        return static_cast<std::size_t>(id);
    }

    static Vertex to_vertex(NodeID id) {
        return Vertex(static_cast<Vertex::ValueType>(id));
    }

    static NodeID to_node(Vertex v) {
        return static_cast<NodeID>(static_cast<Vertex::ValueType>(v));
    }

    Graph& graph_;
    mutable std::shared_mutex mutex_;

    static constexpr AttributeNameWrapper<META_ATTR> meta_attr_{};
};
