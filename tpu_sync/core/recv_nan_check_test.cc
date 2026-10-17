// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include "tpu_sync/core/recv_nan_check.h"

#include <gtest/gtest.h>

#include <array>
namespace tpu_raiden {
namespace {
TEST(RecvNanCheckTest, ClipsCompactPrefixWithoutReadingRegionGapsOrTail) {
  const std::array<NanCheckRange, 2> ranges = {{{0, 16}, {64, 16}}};
  auto chunks = BuildNanCheckChunks(256, ranges, 24);
  ASSERT_EQ(chunks.size(), 2);
  EXPECT_EQ(chunks[0].word_begin, 64);
  EXPECT_EQ(chunks[0].byte_begin, 256);
  EXPECT_EQ(chunks[0].byte_end, 272);
  EXPECT_EQ(chunks[1].word_begin, 80);
  EXPECT_EQ(chunks[1].byte_begin, 320);
  EXPECT_EQ(chunks[1].byte_end, 328);
  EXPECT_TRUE(BuildNanCheckChunks(0, ranges, 0).empty());
}
TEST(RecvNanCheckTest, KeepsHalfWordEndpointsAndBoundsGatherChunks) {
  const std::array<NanCheckRange, 1> ranges = {{{2, 65536}}};
  auto chunks = BuildNanCheckChunks(0, ranges, 65536);
  ASSERT_EQ(chunks.size(), 3);
  EXPECT_EQ(chunks[0].word_begin, 0);
  EXPECT_EQ(chunks[0].byte_begin, 2);
  EXPECT_EQ(chunks[0].byte_end, 32768);
  EXPECT_EQ(chunks[1].byte_begin, 32768);
  EXPECT_EQ(chunks[1].byte_end, 65536);
  EXPECT_EQ(chunks[2].byte_begin, 65536);
  EXPECT_EQ(chunks[2].byte_end, 65538);
}
TEST(RecvNanCheckTest, DumpsFirstThreeNanHitsPerHundredChecksPerPool) {
  NanDumpWindow a, b;
  for (int i = 1; i <= 303; ++i) {
    EXPECT_EQ(a.OnCheck(true), i % 100 >= 1 && i % 100 <= 3) << i;
  }
  EXPECT_TRUE(b.OnCheck(true));
  NanDumpWindow mixed;
  for (int i = 1; i <= 200; ++i) {
    const bool nan = i >= 98 && i <= 104;
    EXPECT_EQ(mixed.OnCheck(nan), nan && i != 104) << i;
  }
  EXPECT_TRUE(mixed.OnCheck(true));
}
TEST(RecvNanCheckTest, SupportsNumericFormatsWithoutGuessingIntegerCarriers) {
  EXPECT_EQ(NanFormatForDtype("bf16")->exponent_mask, 0x7f80);
  EXPECT_EQ(NanFormatForDtype("f16")->fraction_mask, 0x03ff);
  EXPECT_EQ(NanFormatForDtype("float32")->element_bytes, 4);
  EXPECT_FALSE(NanFormatForDtype("uint8"));
  EXPECT_FALSE(NanFormatForDtype("float8_e4m3fn"));
}
}  // namespace
}  // namespace tpu_raiden
