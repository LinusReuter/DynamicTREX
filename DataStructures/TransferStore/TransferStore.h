#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "ITransferStore.h"

namespace transfer_store_detail {
constexpr std::size_t kCacheLineBytes = 64;

struct SpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    void lock() noexcept {
        while (flag.test_and_set(std::memory_order_acquire)) {
        }
    }
    void unlock() noexcept { flag.clear(std::memory_order_release); }
};

template <typename T, std::size_t InlineN>
struct AdjStorageInlineProbe {
    alignas(T) std::byte inline_bytes[sizeof(T) * InlineN];
    std::vector<T> heap_{};
    std::size_t size_{0};
};

constexpr std::size_t pad_to_multiple(std::size_t size, std::size_t multiple) {
    if (multiple == 0) return 0;
    const std::size_t rem = size % multiple;
    return rem == 0 ? 0 : (multiple - rem);
}

template <typename T, std::size_t InlineN, std::size_t Multiple>
consteval std::size_t padded_probe_size() {
    constexpr std::size_t base = sizeof(AdjStorageInlineProbe<T, InlineN>);
    return base + pad_to_multiple(base, Multiple);
}

template <typename T, std::size_t CacheLineBytes, std::size_t Target, std::size_t N>
consteval std::size_t tuned_inline_count_impl() {
    if constexpr (N == 0) {
        return 0;
    } else if constexpr (padded_probe_size<T, N, CacheLineBytes>() <= Target) {
        return N;
    } else {
        return tuned_inline_count_impl<T, CacheLineBytes, Target, N - 1>();
    }
}

template <typename T, std::size_t CacheLineBytes = kCacheLineBytes,
          std::size_t Lines = 1, std::size_t MaxInline = 8>
consteval std::size_t tuned_inline_count() {
    if constexpr (!std::is_default_constructible_v<T> || MaxInline == 0) {
        return 0;
    }
    constexpr std::size_t target = CacheLineBytes * Lines;
    return tuned_inline_count_impl<T, CacheLineBytes, target, MaxInline>();
}

template <typename NodeID, typename EdgeMeta>
struct OutEdgeProbe {
    NodeID to;
    EdgeMeta meta;
};

template <typename NodeID, typename EdgeMeta,
          std::size_t CacheLineBytes = kCacheLineBytes,
          std::size_t Lines = 1, std::size_t MaxInline = 8>
inline constexpr std::size_t tuned_inline_out_v =
    tuned_inline_count<OutEdgeProbe<NodeID, EdgeMeta>, CacheLineBytes, Lines, MaxInline>();

template <typename NodeID,
          std::size_t CacheLineBytes = kCacheLineBytes,
          std::size_t Lines = 1, std::size_t MaxInline = 8>
inline constexpr std::size_t tuned_inline_in_v =
    tuned_inline_count<NodeID, CacheLineBytes, Lines, MaxInline>();

} // namespace transfer_store_detail

/// Vec-of-vecs TransferStore with optional small-adjacency inline storage (SoA layout).
///
/// Design assumptions (match DynamicTB update phases):
/// - Each node is updated at most once per direction per phase.
/// - No concurrent writers on the same (node, direction) in a phase.
/// - Reads are rare and localized (apply*Diff + domination cleanup).
///
/// Notes:
/// - Adjacency lists are kept sorted and unique.
/// - Batch operations buffer ops and apply at commit to keep spans stable.
/// - Striped locks (power-of-two count) protect mirrored updates to the opposite direction.
/// - allowTemporaryInconsistent(...) is currently ignored: this store always keeps
///   incoming/outgoing synchronized on every operation (batch or non-batch).
///
/// Extension: Small adjacency lists are stored inline to avoid heap allocation
/// and improve cache locality for the vast majority of low-degree nodes.
/// Inline storage is optional and padded/aligned to cache-line multiples.
/// Default is **no inline storage** (InlineOut=0 / InlineIn=0).
/// To enable tuned inline storage, use:
///   InlineOut = transfer_store_detail::tuned_inline_out_v<NodeID, EdgeMeta>
///   InlineIn  = transfer_store_detail::tuned_inline_in_v<NodeID>
/// StripeCount controls the number of lock stripes (must be power of two).

template <typename NodeID, typename EdgeMeta,
          std::size_t InlineOut = 0,
          std::size_t InlineIn = 0,
          std::size_t StripeCount = 1024>
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

    template <typename T, std::size_t InlineN, typename Enable = void>
    class AdjStorage;

    // Inline-capable storage (InlineN > 0)
    template <typename T, std::size_t InlineN>
    class alignas(transfer_store_detail::kCacheLineBytes)
        AdjStorage<T, InlineN, std::enable_if_t<(InlineN > 0)>> {
        static_assert(std::is_default_constructible_v<T>,
                      "AdjStorage requires default-constructible T for inline storage");

    public:
        std::size_t size() const noexcept { return size_; }
        bool empty() const noexcept { return size_ == 0; }

        std::span<const T> span() const noexcept {
            if (size_ == 0) return {};
            return is_small() ? std::span<const T>(inline_.data(), size_)
                              : std::span<const T>(heap_.data(), size_);
        }

        void clear() noexcept {
            size_ = 0;
            heap_.clear();
        }

        void reserve(std::size_t n) {
            if (n > InlineN) {
                heap_.reserve(n);
            }
        }

        std::vector<T> to_vector() const {
            if (is_small()) {
                return std::vector<T>(inline_.begin(), inline_.begin() + size_);
            }
            return heap_;
        }

        void assign_sorted(std::vector<T>&& v) {
            size_ = v.size();
            if (is_small()) {
                // Keep heap_ capacity for future growth; just overwrite inline storage.
                std::copy(v.begin(), v.end(), inline_.begin());
                return;
            }
            heap_ = std::move(v);
        }

    private:
        bool is_small() const noexcept { return size_ <= InlineN; }

        std::array<T, InlineN> inline_{};
        std::vector<T> heap_{};
        std::size_t size_{0};

        static constexpr std::size_t kBaseSize =
            sizeof(transfer_store_detail::AdjStorageInlineProbe<T, InlineN>);
        static constexpr std::size_t kPadBytes =
            transfer_store_detail::pad_to_multiple(kBaseSize, transfer_store_detail::kCacheLineBytes);
        std::array<std::byte, kPadBytes> pad_{};
    };

    // Vector-only storage (InlineN == 0)
    template <typename T>
    class AdjStorage<T, 0, void> {
    public:
        std::size_t size() const noexcept { return heap_.size(); }
        bool empty() const noexcept { return heap_.empty(); }

        std::span<const T> span() const noexcept {
            if (heap_.empty()) return {};
            return std::span<const T>(heap_.data(), heap_.size());
        }

        void clear() noexcept { heap_.clear(); }

        void reserve(std::size_t n) { heap_.reserve(n); }

        std::vector<T> to_vector() const { return heap_; }

        void assign_sorted(std::vector<T>&& v) { heap_ = std::move(v); }

    private:
        std::vector<T> heap_{};
    };

    using OutStorage = AdjStorage<OutEdge, InlineOut>;
    using InStorage = AdjStorage<NodeID, InlineIn>;

    struct BatchOp {
        NodeID other{};
        EdgeMeta meta{}; // used for outgoing adds (and incoming adds if mirrored)
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

    // === Helpers ===

    static void normalize_ops(std::vector<BatchOp>& ops) {
        if (ops.empty()) return;
        std::sort(ops.begin(), ops.end(), [](const auto& a, const auto& b) {
            return a.other < b.other;
        });

        std::size_t w = 0;
        for (std::size_t i = 0; i < ops.size();) {
            std::size_t j = i + 1;
            BatchOp last = ops[i];
            while (j < ops.size() && ops[j].other == ops[i].other) {
                last = ops[j];
                ++j;
            }
            ops[w++] = last;
            i = j;
        }
        ops.resize(w);
    }

    static std::size_t stripe_index(NodeID node) {
        return static_cast<std::size_t>(node) & (StripeCount - 1);
    }

    void apply_ops_outgoing_unlocked(NodeID node, std::vector<BatchOp>& ops) {
        if (ops.empty()) return;
        normalize_ops(ops);

        auto current = out_[node].to_vector();
        std::vector<OutEdge> merged;
        merged.reserve(current.size() + ops.size());

        std::size_t i = 0;
        std::size_t j = 0;
        while (i < current.size() || j < ops.size()) {
            if (j == ops.size() || (i < current.size() && current[i].to < ops[j].other)) {
                merged.push_back(current[i++]);
            } else if (i == current.size() || current[i].to > ops[j].other) {
                if (ops[j].add) {
                    merged.push_back(OutEdge{ops[j].other, ops[j].meta});
                }
                ++j;
            } else {
                // equal
                if (ops[j].add) {
                    // Edge already exists; keep existing meta (add is no-op).
                    merged.push_back(current[i]);
                }
                // remove => skip
                ++i;
                ++j;
            }
        }

        out_[node].assign_sorted(std::move(merged));
    }

    void apply_ops_incoming_unlocked(NodeID node, std::vector<BatchOp>& ops) {
        if (ops.empty()) return;
        normalize_ops(ops);

        auto current = in_[node].to_vector();
        std::vector<NodeID> merged;
        merged.reserve(current.size() + ops.size());

        std::size_t i = 0;
        std::size_t j = 0;
        while (i < current.size() || j < ops.size()) {
            if (j == ops.size() || (i < current.size() && current[i] < ops[j].other)) {
                merged.push_back(current[i++]);
            } else if (i == current.size() || current[i] > ops[j].other) {
                if (ops[j].add) {
                    merged.push_back(ops[j].other);
                }
                ++j;
            } else {
                // equal
                if (ops[j].add) {
                    merged.push_back(current[i]);
                }
                // remove => skip
                ++i;
                ++j;
            }
        }

        in_[node].assign_sorted(std::move(merged));
    }

    void apply_ops_outgoing_locked(NodeID node, std::vector<BatchOp>& ops) {
        auto& lock = out_locks_[stripe_index(node)];
        lock.lock();
        apply_ops_outgoing_unlocked(node, ops);
        lock.unlock();
    }

    void apply_ops_incoming_locked(NodeID node, std::vector<BatchOp>& ops) {
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
            const auto out = out_[from].span();
            for (const auto& e : out) {
                ++in_deg[static_cast<std::size_t>(e.to)];
            }
        }

        std::vector<std::vector<NodeID>> tmp(n);
        for (std::size_t i = 0; i < n; ++i) {
            tmp[i].reserve(in_deg[i]);
        }

        for (NodeID from = NodeID(0); from < static_cast<NodeID>(n); ++from) {
            const auto out = out_[from].span();
            for (const auto& e : out) {
                tmp[static_cast<std::size_t>(e.to)].push_back(from);
            }
        }

        for (std::size_t i = 0; i < n; ++i) {
            in_[static_cast<NodeID>(i)].assign_sorted(std::move(tmp[i]));
        }
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

    outgoing_span outgoing_sorted(NodeID from) const override {
        return out_[from].span();
    }

    incoming_span incoming_sorted(NodeID to) const override {
        return in_[to].span();
    }

    EdgeMeta readEdgeMeta(NodeID from, NodeID to) const override {
        const auto span = out_[from].span();
        auto it = std::lower_bound(span.begin(), span.end(), to,
                                   [](const OutEdge& e, NodeID key) { return e.to < key; });
        if (it == span.end() || it->to != to) {
            return EdgeMeta{};
        }
        return it->meta;
    }

    bool add_edge(NodeID from, NodeID to, const EdgeMeta& meta) override {
        const auto span = out_[from].span();
        auto it = std::lower_bound(span.begin(), span.end(), to,
                                   [](const OutEdge& e, NodeID key) { return e.to < key; });
        if (it != span.end() && it->to == to) return false;

        std::vector<BatchOp> ops;
        ops.push_back(BatchOp{to, meta, true});
        apply_ops_outgoing_unlocked(from, ops);

        std::vector<BatchOp> iops;
        iops.push_back(BatchOp{from, meta, true});
        apply_ops_incoming_locked(to, iops);
        return true;
    }

    bool remove_edge(NodeID from, NodeID to) override {
        const auto span = out_[from].span();
        auto it = std::lower_bound(span.begin(), span.end(), to,
                                   [](const OutEdge& e, NodeID key) { return e.to < key; });
        if (it == span.end() || it->to != to) return false;

        std::vector<BatchOp> ops;
        ops.push_back(BatchOp{to, EdgeMeta{}, false});
        apply_ops_outgoing_unlocked(from, ops);

        std::vector<BatchOp> iops;
        iops.push_back(BatchOp{from, EdgeMeta{}, false});
        apply_ops_incoming_locked(to, iops);
        return true;
    }

    bool update_edge_meta(NodeID from, NodeID to, const EdgeMeta& meta) override {
        auto current = out_[from].to_vector();
        auto it = std::lower_bound(current.begin(), current.end(), to,
                                   [](const OutEdge& e, NodeID key) { return e.to < key; });
        if (it == current.end() || it->to != to) return false;
        it->meta = meta;
        out_[from].assign_sorted(std::move(current));
        return true;
    }

    void clear_outgoing(NodeID from) override {
        out_[from].clear();
        // Expensive path: remove from all incoming lists.
        for (std::size_t i = 0; i < in_.size(); ++i) {
            auto& lock = in_locks_[i & (StripeCount - 1)];
            lock.lock();
            auto vec = in_[i].to_vector();
            vec.erase(std::remove(vec.begin(), vec.end(), from), vec.end());
            in_[i].assign_sorted(std::move(vec));
            lock.unlock();
        }
    }

    void clear_incoming(NodeID to) override {
        in_[to].clear();
        // Expensive path: remove from all outgoing lists.
        for (std::size_t i = 0; i < out_.size(); ++i) {
            auto& lock = out_locks_[i & (StripeCount - 1)];
            lock.lock();
            auto vec = out_[i].to_vector();
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                                     [to](const OutEdge& e) { return e.to == to; }),
                      vec.end());
            out_[i].assign_sorted(std::move(vec));
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

    void add_outgoing_edges_init(NodeID from, std::span<const NodeID> to,
                                 const EdgeMeta& meta) override {
        if (!init_mode_) {
            for (const NodeID dst : to) {
                add_edge(from, dst, meta);
            }
            return;
        }

        if (to.empty()) {
            out_[from].clear();
            return;
        }

        std::vector<OutEdge> edges;
        edges.reserve(to.size());
        for (const NodeID dst : to) {
            edges.push_back(OutEdge{dst, meta});
        }

        const bool sorted = std::is_sorted(to.begin(), to.end());
        const bool unique = sorted && (std::adjacent_find(to.begin(), to.end()) == to.end());
        if (!sorted) {
            std::sort(edges.begin(), edges.end(),
                      [](const OutEdge& a, const OutEdge& b) { return a.to < b.to; });
        }
        if (!unique) {
            auto end = std::unique(edges.begin(), edges.end(),
                                   [](const OutEdge& a, const OutEdge& b) { return a.to == b.to; });
            edges.erase(end, edges.end());
        }

        out_[from].assign_sorted(std::move(edges));
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
                std::vector<BatchOp> mirror;
                mirror.push_back(BatchOp{ctx.node, op.meta, op.add});
                apply_ops_outgoing_locked(op.other, mirror);
            }
        } else {
            apply_ops_outgoing_unlocked(ctx.node, ctx.ops);
            // Mirror outgoing ops into incoming (may be contended).
            for (const auto& op : ctx.ops) {
                std::vector<BatchOp> mirror;
                mirror.push_back(BatchOp{ctx.node, op.meta, op.add});
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

    void reserve_outgoing(NodeID node, std::size_t n) override {
        out_[node].reserve(n);
    }

    void reserve_incoming(NodeID node, std::size_t n) override {
        in_[node].reserve(n);
    }

private:
    std::array<transfer_store_detail::SpinLock, StripeCount> out_locks_{};
    std::array<transfer_store_detail::SpinLock, StripeCount> in_locks_{};
    std::vector<OutStorage> out_{};
    std::vector<InStorage> in_{};
    bool allow_inconsistent_requested_{false};
    bool init_mode_{false};
};
