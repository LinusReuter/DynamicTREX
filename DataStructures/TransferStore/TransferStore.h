#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

#include "ITransferStore.h"

// Maybe test out small vec as alternative e.g.
// #include "gch/small_vector.hpp"

namespace transfer_store_detail {
struct SpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    void lock() noexcept {
        while (flag.test_and_set(std::memory_order_acquire)) {}
    }
    void unlock() noexcept { flag.clear(std::memory_order_release); }
};

// Utility for O(1) unordered erasure
template <typename Vec, typename Pred>
bool swap_erase_if(Vec& v, Pred p) {
    auto it = std::find_if(v.begin(), v.end(), p);
    if (it != v.end()) {
        *it = std::move(v.back());
        v.pop_back();
        return true;
    }
    return false;
}

} // namespace transfer_store_detail

template <typename NodeID, typename EdgeMeta, std::size_t StripeCount = 1024>
class TransferStore final : public ITransferStore<NodeID, EdgeMeta> {
public:
    using Base = ITransferStore<NodeID, EdgeMeta>;
    using OutEdge = typename Base::OutEdge;
    using outgoing_span = typename Base::outgoing_span;
    using incoming_span = typename Base::incoming_span;
    using batch_id_type = typename Base::batch_id_type;
    using Direction = typename Base::Direction;

private:
    static_assert(StripeCount > 0, "StripeCount must be > 0");
    static_assert((StripeCount & (StripeCount - 1)) == 0, "StripeCount must be a power of two");

    // Possible change to SmallVec
    // Example: using OutStorage = gch::small_vector<OutEdge, 8>;
    using OutStorage = std::vector<OutEdge>;
    using InStorage  = std::vector<NodeID>;

    struct BatchOp {
        NodeID other{};
        EdgeMeta meta{};
        bool add{false};
    };

    struct BatchContext {
        NodeID node{};
        bool incoming{false};
        std::vector<BatchOp> ops;
        void clear() { ops.clear(); }
    };

    static BatchContext& batch_context() {
        thread_local BatchContext ctx{};
        return ctx;
    }

    static std::size_t stripe_index(NodeID node) {
        return static_cast<std::size_t>(node) & (StripeCount - 1);
    }

    void apply_ops_outgoing_unlocked(NodeID node, const std::vector<BatchOp>& ops) {
        if (ops.empty()) return;

        std::size_t adds = 0;
        for (const auto& op : ops) {
            adds = (op.add) ? adds++ : adds--;
        }

        auto& storage = out_[node];
        storage.reserve(storage.size() + adds);

        for (const auto& op : ops) {
            auto it = std::find_if(storage.begin(), storage.end(),
                                   [&](const OutEdge& e) { return e.to == op.other; });
            if (op.add) {
                if (it != storage.end()) {
                    it->meta = op.meta; // Edge exists, update meta
                } else {
                    storage.push_back(OutEdge{op.other, op.meta});
                }
            } else {
                if (it != storage.end()) {
                    *it = std::move(storage.back());
                    storage.pop_back();
                }
            }
        }
    }

    void apply_ops_incoming_unlocked(NodeID node, const std::vector<BatchOp>& ops) {
        if (ops.empty()) return;

        std::size_t adds = 0;
        for (const auto& op : ops) {
            adds = (op.add) ? adds++ : adds--;
        }

        auto& storage = in_[node];
        storage.reserve(storage.size() + adds);

        for (const auto& op : ops) {
            if (op.add) {
                auto it = std::find(storage.begin(), storage.end(), op.other);
                if (it == storage.end()) {
                    storage.push_back(op.other);
                }
            } else {
                transfer_store_detail::swap_erase_if(storage, [&](NodeID n){ return n == op.other; });
            }
        }
    }

    void apply_ops_outgoing_locked(NodeID node, const std::vector<BatchOp>& ops) {
        auto& lock = out_locks_[stripe_index(node)];
        lock.lock();
        apply_ops_outgoing_unlocked(node, ops);
        lock.unlock();
    }

    void apply_ops_incoming_locked(NodeID node, const std::vector<BatchOp>& ops) {
        auto& lock = in_locks_[stripe_index(node)];
        lock.lock();
        apply_ops_incoming_unlocked(node, ops);
        lock.unlock();
    }

    // Build incoming from outgoing (sorted by from).
    void rebuild_incoming_from_outgoing() {
        const std::size_t n = out_.size();
        std::vector<std::size_t> in_deg(n, 0);

        for (NodeID from = NodeID(0); from < static_cast<NodeID>(n); ++from) {
            for (const auto& e : out_[from]) {
                ++in_deg[static_cast<std::size_t>(e.to)];
            }
        }

        std::vector<InStorage> tmp(n);
        for (std::size_t i = 0; i < n; ++i) {
            tmp[i].reserve(in_deg[i]);
        }

        for (NodeID from = NodeID(0); from < static_cast<NodeID>(n); ++from) {
            for (const auto& e : out_[from]) {
                tmp[static_cast<std::size_t>(e.to)].push_back(from);
            }
        }

        in_ = std::move(tmp);
    }

public:
    std::size_t node_count() const noexcept override { return out_.size(); }

    void add_nodes(NodeID maxNodeId) override {
        const std::size_t target = static_cast<std::size_t>(maxNodeId) + 1;
        if (target > out_.size()) {
            out_.resize(target);
            in_.resize(target);
        }
    }

    // --- Explicit Sorting Methods ---
    void sort_outgoing(const NodeID from) const {
        auto span = out_[from];

        // Get non-const pointers to the underlying data
        auto* first = const_cast<OutEdge*>(span.data());
        auto* last = first + span.size();

        std::sort(first, last, [](const OutEdge& a, const OutEdge& b) {
            return a.to < b.to;
        });
    }

    void sort_incoming(const NodeID to) const {
        auto span = in_[to];

        // Get non-const pointers to the underlying data
        auto* first = const_cast<NodeID*>(span.data());
        auto* last = first + span.size();

        std::sort(first, last);
    }

    // Spans are now inherently unordered unless explicit sort is called
    outgoing_span outgoing_sorted(const NodeID from) const override {
        sort_outgoing(from);
        return {out_[from].data(), out_[from].size()};
    }

    incoming_span incoming_sorted(const NodeID to) const override {
        sort_incoming(to);
        return {in_[to].data(), in_[to].size()};
    }

    EdgeMeta readEdgeMeta(NodeID from, NodeID to) const override {
        const auto& storage = out_[from];
        auto it = std::find_if(storage.begin(), storage.end(),
                               [to](const OutEdge& e) { return e.to == to; });
        return (it != storage.end()) ? it->meta : EdgeMeta{};
    }

    bool add_edge(NodeID from, NodeID to, const EdgeMeta& meta) override {
        auto& storage = out_[from];
        auto it = std::find_if(storage.begin(), storage.end(),
                               [to](const OutEdge& e) { return e.to == to; });
        if (it != storage.end()) return false;

        storage.push_back(OutEdge{to, meta});

        std::vector<BatchOp> iops = {{from, meta, true}};
        apply_ops_incoming_locked(to, iops);
        return true;
    }

    bool remove_edge(NodeID from, NodeID to) override {
        if (!transfer_store_detail::swap_erase_if(out_[from], [to](const OutEdge& e) { return e.to == to; })) {
            return false;
        }

        std::vector<BatchOp> iops = {{from, EdgeMeta{}, false}};
        apply_ops_incoming_locked(to, iops);
        return true;
    }

    bool update_edge_meta(NodeID from, NodeID to, const EdgeMeta& meta) override {
        auto& storage = out_[from];
        auto it = std::find_if(storage.begin(), storage.end(),
                               [to](const OutEdge& e) { return e.to == to; });
        if (it == storage.end()) return false;

        it->meta = meta;
        return true;
    }

    void clear_outgoing(NodeID from) override {
        // Fast path: trace edges rather than scanning V locks
        auto out_edges = std::move(out_[from]);
        out_[from].clear(); // out_edges now owns the memory for this traversal

        for (const auto& e : out_edges) {
            auto& lock = in_locks_[stripe_index(e.to)];
            lock.lock();
            transfer_store_detail::swap_erase_if(in_[e.to], [from](NodeID n){ return n == from; });
            lock.unlock();
        }
    }

    void clear_incoming(NodeID to) override {
        auto in_edges = std::move(in_[to]);
        in_[to].clear();

        for (const NodeID from : in_edges) {
            auto& lock = out_locks_[stripe_index(from)];
            lock.lock();
            transfer_store_detail::swap_erase_if(out_[from], [to](const OutEdge& e){ return e.to == to; });
            lock.unlock();
        }
    }

    void clear() override {
        out_.clear();
        in_.clear();
    }

    void allowTemporaryInconsistent(bool allow) override { allow_inconsistent_requested_ = allow; }

    // === Initial build helpers (outgoing-only, incoming built in bulk) ===
    void begin_outgoing_init(NodeID maxNodeId) override {
        add_nodes(maxNodeId);
        allowTemporaryInconsistent(true);
        init_mode_ = true;
    }

    void add_outgoing_edge_init(NodeID from, NodeID to, const EdgeMeta& meta) override {
        if (!init_mode_) {
            add_edge(from, to, meta);
            return;
        }
        std::array<NodeID, 1> one{to};
        add_outgoing_edges_init(from, std::span<const NodeID>(one), meta);
    }

    void add_outgoing_edges_init(NodeID from, std::span<const NodeID> to, const EdgeMeta& meta) override {
        if (!init_mode_) {
            for (const NodeID dst : to) add_edge(from, dst, meta);
            return;
        }

        auto& storage = out_[from];
        storage.reserve(storage.size() + to.size());

        for (const NodeID dst : to) {
            // In init mode, we assume unique appends (caller responsibility to deduplicate if needed)
            storage.push_back(OutEdge{dst, meta});
        }
    }

    void finish_outgoing_init() override {
        rebuild_incoming_from_outgoing();
        sync_barrier();
        allowTemporaryInconsistent(false);
        init_mode_ = false;
    }

    void rebuild_incoming() override { rebuild_incoming_from_outgoing(); }

    void sync_barrier() override {
        // No-op for now: this store maintains full consistency eagerly.
        // Future async reconciliation can use allow_inconsistent_requested_.
    }

    // Batch operations assume the caller does not enqueue duplicate add/remove for the same edge
    // within a batch (true for apply*Diff). Return values indicate enqueue success only.
    batch_id_type begin_batch(NodeID node, Direction dir) override {
        auto& ctx = batch_context();
        ctx.node = node;
        ctx.incoming = (dir == Direction::Incoming);
        ctx.clear();
        return batch_id_type{0};
    }

    void commit_batch(batch_id_type) override {
        auto& ctx = batch_context();
        if (ctx.incoming) {
            apply_ops_incoming_unlocked(ctx.node, ctx.ops);
            // Mirror incoming ops into outgoing (may be contended).
            for (const auto& op : ctx.ops) {
                std::vector<BatchOp> mirror = {{ctx.node, op.meta, op.add}};
                apply_ops_outgoing_locked(op.other, mirror);
            }
        } else {
            apply_ops_outgoing_unlocked(ctx.node, ctx.ops);
            // Mirror outgoing ops into incoming (may be contended).
            for (const auto& op : ctx.ops) {
                std::vector<BatchOp> mirror = {{ctx.node, op.meta, op.add}};
                apply_ops_incoming_locked(op.other, mirror);
            }
        }
        ctx.clear();
    }

    bool add_outgoing_edge(batch_id_type, NodeID to, const EdgeMeta& meta) override {
        auto& ctx = batch_context();
        if (ctx.incoming) return false;
        ctx.ops.push_back(BatchOp{to, meta, true});
        return true;
    }

    bool remove_outgoing_edge(batch_id_type, NodeID to) override {
        auto& ctx = batch_context();
        if (ctx.incoming) return false;
        ctx.ops.push_back(BatchOp{to, EdgeMeta{}, false});
        return true;
    }

    bool add_incoming_edge(batch_id_type, NodeID from, const EdgeMeta& meta) override {
        auto& ctx = batch_context();
        if (!ctx.incoming) return false;
        ctx.ops.push_back(BatchOp{from, meta, true});
        return true;
    }

    bool remove_incoming_edge(batch_id_type, NodeID from) override {
        auto& ctx = batch_context();
        if (!ctx.incoming) return false;
        ctx.ops.push_back(BatchOp{from, EdgeMeta{}, false});
        return true;
    }

    void reserve_outgoing(NodeID node, std::size_t n) override { out_[node].reserve(n); }
    void reserve_incoming(NodeID node, std::size_t n) override { in_[node].reserve(n); }

private:
    std::array<transfer_store_detail::SpinLock, StripeCount> out_locks_{};
    std::array<transfer_store_detail::SpinLock, StripeCount> in_locks_{};
    std::vector<OutStorage> out_{};
    std::vector<InStorage> in_{};
    bool allow_inconsistent_requested_{false};
    bool init_mode_{false};
};
