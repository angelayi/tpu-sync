// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cassert>
#include <limits>

#include "tpu_sync/core/recv_nan_check.h"
namespace tpu_raiden {
std::vector<NanCheckChunk> BuildNanCheckChunks(
    int64_t block_base, std::span<const NanCheckRange> ranges,
    int64_t live_bytes) {
  std::vector<NanCheckChunk> chunks;
  assert(block_base >= 0 && live_bytes >= 0);
  for (const auto& range : ranges) {
    if (live_bytes == 0) break;
    const int64_t size = std::min(range.size, live_bytes);
    assert(range.offset >= 0 && size >= 0 &&
           block_base <= std::numeric_limits<int64_t>::max() - range.offset);
    int64_t begin = block_base + range.offset;
    assert(begin <= std::numeric_limits<int64_t>::max() - size);
    const int64_t end = begin + size;
    while (begin < end) {
      const int64_t word_begin = begin / 4;
      const int64_t count =
          std::min(end - begin, kNanCheckChunkBytes - (begin % 4));
      chunks.push_back({word_begin, begin, begin + count});
      begin += count;
    }
    live_bytes -= size;
  }
  assert(live_bytes == 0 && "received live prefix exceeds pool live regions");
  return chunks;
}
}  // namespace tpu_raiden
