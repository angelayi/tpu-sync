// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#ifndef TPU_SYNC_CORE_RECV_NAN_CHECK_H_
#define TPU_SYNC_CORE_RECV_NAN_CHECK_H_
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <vector>
namespace tpu_sync::rpc {
class StartTransferRequest;
}
namespace tpu_raiden {
namespace kv_cache {
class KVCacheManagerBase;
}
struct NanFormat {
  size_t element_bytes;
  uint32_t exponent_mask;
  uint32_t fraction_mask;
};
inline std::optional<NanFormat> NanFormatForDtype(std::string_view dtype) {
  if (dtype == "bfloat16" || dtype == "bf16")
    return NanFormat{2, 0x7f80, 0x007f};
  if (dtype == "float16" || dtype == "f16") return NanFormat{2, 0x7c00, 0x03ff};
  if (dtype == "float32" || dtype == "f32")
    return NanFormat{4, 0x7f800000, 0x007fffff};
  return std::nullopt;
}
struct NanCheckRange {
  int64_t offset;
  int64_t size;
};
// Absolute physical byte intervals split into bounded gather/dump chunks.
struct NanCheckChunk {
  int64_t word_begin;
  int64_t byte_begin;
  int64_t byte_end;
};
inline constexpr int64_t kNanCheckChunkBytes = 32768;
std::vector<NanCheckChunk> BuildNanCheckChunks(
    int64_t block_base, std::span<const NanCheckRange> ranges,
    int64_t live_bytes);
// Fixed windows of 100 successful receive checks, owned by one manager/pool.
class NanDumpWindow {
 public:
  bool OnCheck(bool has_nan) {
    if (checks_ == 100) {
      checks_ = 0;
      dumps_ = 0;
    }
    ++checks_;
    if (!has_nan || dumps_ == 3) return false;
    ++dumps_;
    return true;
  }

 private:
  uint32_t checks_ = 0;
  uint32_t dumps_ = 0;
};
// Owns native PJRT programs and per-pool dump windows for one manager.
class RecvNanCheck {
 public:
  // A weak registration leaves disabled-mode buffer lifetime unchanged.
  explicit RecvNanCheck(std::weak_ptr<void> inputs_owner);
  class Task {
   public:
    ~Task();
    // Off the DMA callback thread, after ALL H2D futures completed. Bool
    // readback and admitted KV dumps finish before this returns.
    void Run();
    void Schedule(std::function<void()> completion);

   private:
    friend class RecvNanCheck;
    struct Impl;
    explicit Task(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };
  static std::unique_ptr<Task> Create(
      kv_cache::KVCacheManagerBase* base,
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      std::shared_ptr<RecvNanCheck>* manager_owner);
  ~RecvNanCheck();
  // The manager calls this after draining sessions, before device teardown.
  void Drain();

 private:
  struct Impl;
  std::weak_ptr<void> inputs_owner_;
  std::mutex init_mu_;
  std::unique_ptr<Impl> impl_;
};
}  // namespace tpu_raiden
#endif
