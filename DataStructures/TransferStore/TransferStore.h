#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>
#include <iostream>
#include <utility>

#include "ITransferStore.h"
#include "ExternalLibs/gch_small_vector/small_vector.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define SPINLOCK_PAUSE() _mm_pause()
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define SPINLOCK_PAUSE() asm volatile("yield" ::: "memory")
#else
#define SPINLOCK_PAUSE() ((void)0)
#endif

static constexpr bool logging = false;

namespace transfer_store_detail {
template<bool ThreadSafe>
struct alignas(std::hardware_destructive_interference_size) SpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    void lock() noexcept {
        if constexpr (ThreadSafe) {
            while (flag.test_and_set(std::memory_order_acquire)) {
                while (flag.test(std::memory_order_relaxed)) {
                    SPINLOCK_PAUSE();
                }
            }
        }
    }
    void unlock() noexcept {
        if constexpr (ThreadSafe) {
            flag.clear(std::memory_order_release);
        }
    }
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

template <typename NodeID, typename EdgeMeta, std::size_t StripeCount = 256, bool ThreadSafe = true>
class TransferStore final : public ITransferStore<NodeID, EdgeMeta> {
public:
    using Base = ITransferStore<NodeID, EdgeMeta>;
    using OutEdge = typename Base::OutEdge;
    using outgoing_span = typename Base::outgoing_span;
    using incoming_span = typename Base::incoming_span;
    using batch_id_type = typename Base::batch_id_type;
    using Direction = typename Base::Direction;

    TransferStore() = default;

    explicit TransferStore(const std::string& fileName) { deserialize(fileName); }

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
        gch::small_vector<BatchOp, 16> ops; // Keeps batches of <= 16 operations entirely on the stack
        void clear() { ops.clear(); }
    };

    static BatchContext& batch_context() {
        thread_local BatchContext ctx{};
        return ctx;
    }

    static std::size_t stripe_index(NodeID node) {
        return static_cast<std::size_t>(node) & (StripeCount - 1);
    }

    void apply_ops_outgoing_unlocked(NodeID node, std::span<const BatchOp> ops) {
        if (ops.empty()) return;

        auto& storage = out_[node];

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

    void apply_ops_incoming_unlocked(NodeID node, std::span<const BatchOp> ops) {
        if (ops.empty()) return;

        auto& storage = in_[node];

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

    void apply_ops_outgoing_locked(NodeID node, std::span<const BatchOp> ops) {
        auto& lock = out_locks_[stripe_index(node)];
        lock.lock();
        apply_ops_outgoing_unlocked(node, ops);
        lock.unlock();
    }

    void apply_ops_incoming_locked(NodeID node, std::span<const BatchOp> ops) {
        auto& lock = in_locks_[stripe_index(node)];
        lock.lock();
        apply_ops_incoming_unlocked(node, ops);
        lock.unlock();
    }

    void apply_op_outgoing_locked(NodeID node, const BatchOp& op) {
        auto& lock = out_locks_[stripe_index(node)];
        lock.lock();
        apply_op_outgoing_unlocked(node, op);
        lock.unlock();
    }

    void apply_op_outgoing_unlocked(NodeID node, const BatchOp& op) {
        auto& storage = out_[node];
        if (op.add) {
            auto it = std::find_if(storage.begin(), storage.end(),
                                   [&](const OutEdge& e) { return e.to == op.other; });
            if (it != storage.end()) {
                it->meta = op.meta; // Edge exists, update meta
            } else {
                storage.push_back(OutEdge{op.other, op.meta});
            }
        } else {
            transfer_store_detail::swap_erase_if(storage, [&](const OutEdge& e) { return e.to == op.other; });
        }
    }

    void apply_op_incoming_locked(NodeID node, const BatchOp& op) {
        auto& lock = out_locks_[stripe_index(node)];
        lock.lock();
        apply_op_outgoing_unlocked(node, op);
        lock.unlock();
    }

    void apply_op_incoming_unlocked(NodeID node, const BatchOp& op) {
        auto& storage = in_[node];
        if (op.add) {
            auto it = std::find(storage.begin(), storage.end(), op.other);
            if (it == storage.end()) {
                storage.push_back(op.other);
            } else {
                it->meta = op.meta; // Edge exists, update meta
            }
        } else {
            transfer_store_detail::swap_erase_if(storage, [&](NodeID n){ return n == op.other; });
        }
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
    void sort_outgoing(const NodeID from) {
        OutStorage& storage = out_[from];
        std::sort(storage.begin(), storage.end(), [](OutEdge& a, OutEdge& b) {
            return a.to < b.to;
        });
    }

    void sort_incoming(const NodeID to) {
        InStorage& storage = in_[to];
        std::sort(storage.begin(), storage.end());
    }

    outgoing_span outgoing_unsorted(const NodeID from) override {
        return {out_[from].data(), out_[from].size()};
    }

    incoming_span incoming_unsorted(const NodeID to) override {
        return {in_[to].data(), in_[to].size()};
    }

    // Snapshot outgoing adjacency under the outgoing stripe lock. Safe to iterate
    // even while other threads mirror incoming edges into this node's outgoing list
    // (which may reallocate out_[from] and invalidate any span returned above).
    void copy_outgoing(const NodeID from, std::vector<OutEdge>& out) override {
        auto& lock = out_locks_[stripe_index(from)];
        lock.lock();
        const auto& storage = out_[from];
        out.assign(storage.begin(), storage.end());
        lock.unlock();
    }

    outgoing_span outgoing_sorted(const NodeID from) override {
        sort_outgoing(from);
        return {out_[from].data(), out_[from].size()};
    }

    incoming_span incoming_sorted(const NodeID to) override {
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
        log_modification(from, to, true);
        auto& lockF = out_locks_[stripe_index(from)];
        auto& lockT = in_locks_[stripe_index(to)];
        lockF.lock();
        auto& out = out_[from];
        auto it = std::find_if(out.begin(), out.end(),
                               [to](const OutEdge& e) { return e.to == to; });
        if (it != out.end()) {
            lockF.unlock();
            return false;
        }
        out.push_back(OutEdge{to, meta});
        lockF.unlock();
        lockT.lock();
        auto& in = in_[to];
        auto it2 = std::find_if(in.begin(), in.end(), [from](const NodeID e) { return e == from; });
        if (it2 == in.end()) in.push_back(from);
        lockT.unlock();
        return true;
    }

    bool remove_edge(NodeID from, NodeID to) override {
        log_modification(from, to, false);
        auto& lockF = out_locks_[stripe_index(from)];
        auto& lockT = in_locks_[stripe_index(to)];
        lockF.lock();
        const bool erased =
            transfer_store_detail::swap_erase_if(out_[from], [to](const OutEdge& e) { return e.to == to; });
        lockF.unlock();
        if (!erased) {
            return false;
        }
        lockT.lock();
        transfer_store_detail::swap_erase_if(in_[to], [&](NodeID n){ return n == from; });
        lockT.unlock();
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
        for (const auto& e : out_[from]) {
            auto& lock = in_locks_[stripe_index(e.to)];
            lock.lock();
            transfer_store_detail::swap_erase_if(in_[e.to], [from](NodeID n){ return n == from; });
            lock.unlock();
        }
        out_[from].clear();
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

        log("Batch BEGIN node=",
            node,
            " dir=",
            (ctx.incoming ? "incoming" : "outgoing"));

        return batch_id_type{0};
    }

    void commit_batch(batch_id_type) override {
        auto& ctx = batch_context();

        log("Batch COMMIT node=",
            ctx.node,
            " dir=",
            (ctx.incoming ? "incoming" : "outgoing"),
            " ops=",
            ctx.ops.size());

        if (ctx.incoming) {
            apply_ops_incoming_unlocked(ctx.node, ctx.ops);
            // Mirror incoming ops into outgoing (may be contended).
            for (const auto& op : ctx.ops) {
                log_modification(op.other, ctx.node, op.add);

                std::array<BatchOp, 1> mirror{BatchOp{ctx.node, op.meta, op.add}};
                apply_ops_outgoing_locked(op.other, mirror);
            }
        } else {
            apply_ops_outgoing_unlocked(ctx.node, ctx.ops);
            // Mirror outgoing ops into incoming (may be contended).
            for (const auto& op : ctx.ops) {
                log_modification(ctx.node, op.other, op.add);

                std::array<BatchOp, 1> mirror{BatchOp{ctx.node, op.meta, op.add}};
                apply_ops_incoming_locked(op.other, mirror);
            }
        }

        log("Batch END node=",
            ctx.node,
            " applied_ops=",
            ctx.ops.size());

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

    int out_degree(NodeID node_id) override {
        return out_[node_id].size();
    }

    template<typename... Args>
    static void log(Args&&... args) {
        if constexpr (logging) {
            (std::cout << ... << std::forward<Args>(args)) << '\n';
        }
    }

    static void log_modification(NodeID from, NodeID to, bool addition) {
        if constexpr (logging) {
            log("Edge ",
                (addition ? "ADD" : "REMOVE"),
                " ",
                from,
                " -> ",
                to);
        }
    }

    void serialize(const std::string& fileName) const noexcept {
        IO::serialize(fileName, out_, in_);
    }

    void deserialize(const std::string& fileName) noexcept {
        IO::deserialize(fileName, out_, in_);
    }

    std::unordered_map<u_int64_t, u_int64_t> edgeDegreeDistrebutionOut() const noexcept override {
        std::unordered_map<u_int64_t, u_int64_t> degreeDist;
        for (const auto& node : out_) {
            ++degreeDist[node.size()];
        }
        return degreeDist;
    }

    std::unordered_map<u_int64_t, u_int64_t> edgeDegreeDistrebutionIn() const noexcept override {
        std::unordered_map<u_int64_t, u_int64_t> degreeDist;
        for (const auto& node : in_) {
            ++degreeDist[node.size()];
        }
        return degreeDist;
    }

private:
    std::array<transfer_store_detail::SpinLock<ThreadSafe>, StripeCount> out_locks_{};
    std::array<transfer_store_detail::SpinLock<ThreadSafe>, StripeCount> in_locks_{};
    std::vector<OutStorage> out_{};
    std::vector<InStorage> in_{};
    bool allow_inconsistent_requested_{false};
    bool init_mode_{false};
};
