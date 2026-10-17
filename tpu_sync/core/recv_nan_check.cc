// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include "tpu_sync/core/recv_nan_check.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/shape_util.h"

namespace tpu_raiden {
namespace {
bool IsFullAttention(std::string_view tag) {
  if (tag == "fa") return true;
  return tag.starts_with("fa.l") && tag.size() > 4 &&
         std::all_of(tag.begin() + 4, tag.end(),
                     [](char ch) { return ch >= '0' && ch <= '9'; });
}
struct ReceivedBlock {
  int64_t id;
  std::vector<NanCheckChunk> chunks;
};
struct ReceivedPool {
  size_t index;
  std::string tag;
  NanFormat format;
  std::vector<ReceivedBlock> blocks;
  std::vector<raiden::BufferHoldAndAlias> shards;
};
struct ShardInput {
  const ReceivedPool* pool;
  size_t shard;
  const raiden::BufferHoldAndAlias* hold;
  std::vector<NanCheckChunk> chunks;
  int64_t bucket = 1;
};
struct BufferDeleter {
  const PJRT_Api* api;
  void operator()(PJRT_Buffer* buffer) const {
    PJRT_Buffer_Destroy_Args args{};
    args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    args.buffer = buffer;
    raiden::PjrtErrorToStatusLocal(api, api->PJRT_Buffer_Destroy(&args))
        .IgnoreError();
  }
};
using Buffer = std::unique_ptr<PJRT_Buffer, BufferDeleter>;
struct Program {
  const PJRT_Api* api;
  PJRT_LoadedExecutable* executable;
  ~Program() {
    PJRT_LoadedExecutable_Destroy_Args args{};
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.executable = executable;
    raiden::PjrtErrorToStatusLocal(api,
                                   api->PJRT_LoadedExecutable_Destroy(&args))
        .IgnoreError();
  }
};

absl::StatusOr<xla::Shape> BufferShape(const PJRT_Api* api, PJRT_Buffer* buffer,
                                       xla::PrimitiveType type,
                                       std::initializer_list<int64_t> dims) {
  PJRT_Buffer_GetMemoryLayout_Args args{};
  args.struct_size = PJRT_Buffer_GetMemoryLayout_Args_STRUCT_SIZE;
  args.buffer = buffer;
  auto status = raiden::PjrtErrorToStatusLocal(
      api, api->PJRT_Buffer_GetMemoryLayout(&args));
  if (!status.ok()) return status;
  if (args.layout.type != PJRT_Buffer_MemoryLayout_Type_Tiled)
    return absl::UnimplementedError(
        "FA diagnostic requires tiled PJRT device layout");
  auto proto = xla::ShapeUtil::MakeShape(type, dims).ToProto();
  auto* layout = proto.mutable_layout();
  layout->clear_minor_to_major();
  const auto& tiled = args.layout.tiled;
  for (size_t i = 0; i < tiled.minor_to_major_size; ++i)
    layout->add_minor_to_major(tiled.minor_to_major[i]);
  size_t offset = 0;
  for (size_t i = 0; i < tiled.num_tiles; ++i) {
    auto* tile = layout->add_tiles();
    for (size_t j = 0; j < tiled.tile_dim_sizes[i]; ++j)
      tile->add_dimensions(tiled.tile_dims[offset++]);
  }
  return xla::Shape::FromProto(proto);
}

xla::XlaOp ScalarU32(xla::XlaBuilder* b, uint32_t v) {
  return xla::ConstantR0<uint32_t>(b, v);
}
xla::XlaOp ScalarS64(xla::XlaBuilder* b, int64_t v) {
  return xla::ConstantR0<int64_t>(b, v);
}

absl::StatusOr<std::shared_ptr<Program>> Compile(
    const PJRT_Api* api, PJRT_Client* client,
    const std::vector<ShardInput>& inputs,
    const std::vector<xla::Shape>& shapes) {
  xla::XlaBuilder b("received_fa_nan");
  xla::XlaBuilder rb("received_fa_nan_or");
  auto a =
      xla::Parameter(&rb, 0, xla::ShapeUtil::MakeShape(xla::PRED, {}), "a");
  auto c =
      xla::Parameter(&rb, 1, xla::ShapeUtil::MakeShape(xla::PRED, {}), "b");
  auto reduce_or = rb.Build(xla::Or(a, c));
  if (!reduce_or.ok()) return reduce_or.status();
  auto meta =
      xla::Parameter(&b, inputs.size(), shapes.back(), "received_byte_ranges");
  std::vector<xla::XlaOp> flags;
  int64_t row = 0;
  constexpr int64_t words = kNanCheckChunkBytes / 4;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& in = inputs[i];
    auto raw = xla::Parameter(&b, i, shapes[i], "raw_kv_words");
    auto column = [&](int col) {
      auto x = xla::Slice(meta, {row, col}, {row + in.bucket, col + 1}, {1, 1});
      return xla::BroadcastInDim(xla::Reshape(x, {in.bucket}),
                                 {in.bucket, words}, {0});
    };
    auto word_index = xla::Add(
        column(0),
        xla::Iota(&b, xla::ShapeUtil::MakeShape(xla::S64, {in.bucket, words}),
                  1));
    xla::GatherDimensionNumbers dims;
    dims.add_collapsed_slice_dims(0);
    dims.add_start_index_map(0);
    dims.set_index_vector_dim(2);  // implicit scalar index for every word
    auto bits = xla::Gather(raw, word_index, dims, {1});
    auto byte = xla::Mul(word_index, ScalarS64(&b, 4));
    auto begin = column(1), end = column(2);
    const auto& f = in.pool->format;
    auto is_nan = [&](xla::XlaOp value) {
      return xla::And(xla::Eq(xla::And(value, ScalarU32(&b, f.exponent_mask)),
                              ScalarU32(&b, f.exponent_mask)),
                      xla::Ne(xla::And(value, ScalarU32(&b, f.fraction_mask)),
                              ScalarU32(&b, 0)));
    };
    auto valid = [&](xla::XlaOp start, int64_t bytes) {
      return xla::And(xla::Ge(start, begin),
                      xla::Le(xla::Add(start, ScalarS64(&b, bytes)), end));
    };
    auto bad = xla::And(is_nan(bits), valid(byte, f.element_bytes));
    if (f.element_bytes == 2) {
      auto high = xla::ShiftRightLogical(bits, ScalarU32(&b, 16));
      bad = xla::Or(bad, xla::And(is_nan(high),
                                  valid(xla::Add(byte, ScalarS64(&b, 2)), 2)));
    }
    auto flag =
        xla::Reduce(bad, xla::ConstantR0<bool>(&b, false), *reduce_or, {0, 1});
    flags.push_back(xla::Reshape(flag, {1}));
    row += in.bucket;
  }
  auto computation = b.Build(xla::ConcatInDim(&b, flags, 0));
  if (!computation.ok()) return computation.status();
  xla::CompileOptionsProto options;
  options.set_compile_portable_executable(true);
  for (const auto& shape : shapes)
    *options.add_argument_layouts() = shape.ToProto();
  auto* build = options.mutable_executable_build_options();
  build->set_device_ordinal(-1);
  build->set_num_replicas(1);
  build->set_num_partitions(1);
  *build->mutable_debug_options() = xla::DefaultDebugOptionsIgnoringFlags();
  std::string code = computation->proto().SerializeAsString();
  const std::string serialized = options.SerializeAsString();
  PJRT_Program program{};
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.code = code.data();
  program.code_size = code.size();
  program.format = "hlo";
  program.format_size = 3;
  PJRT_Client_Compile_Args compile{};
  compile.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  compile.client = client;
  compile.program = &program;
  compile.compile_options = serialized.data();
  compile.compile_options_size = serialized.size();
  auto status =
      raiden::PjrtErrorToStatusLocal(api, api->PJRT_Client_Compile(&compile));
  if (!status.ok()) return status;
  return std::shared_ptr<Program>(new Program{api, compile.executable});
}

float Decode(const std::byte* p, NanFormat f) {
  uint32_t bits = 0;
  std::memcpy(&bits, p, f.element_bytes);
  if (f.element_bytes == 4) return std::bit_cast<float>(bits);
  if (f.exponent_mask == 0x7f80) return std::bit_cast<float>(bits << 16);
  const uint32_t e = (bits >> 10) & 31, m = bits & 1023;
  const float value = e == 31 ? (m ? std::numeric_limits<float>::quiet_NaN()
                                   : std::numeric_limits<float>::infinity())
                              : std::ldexp(static_cast<float>(e ? 1024 + m : m),
                                           e ? int(e) - 25 : -24);
  return bits & 0x8000 ? -value : value;
}

absl::Status DumpPool(const ReceivedPool& pool, uint64_t uuid,
                      const std::string& req) {
  LOG(ERROR) << "KV_NAN_DUMP_BEGIN req_id=" << req << " uuid=" << uuid
             << " pool=" << pool.index << " tag=" << pool.tag
             << " source=device_hbm";
  size_t values = 0;
  for (size_t shard = 0; shard < pool.shards.size(); ++shard) {
    const auto& hold = pool.shards[shard];
    for (const auto& block : pool.blocks)
      for (const auto& chunk : block.chunks) {
        // Independent storage, never the staging arena. Aligned raw reads also
        // work for half-word endpoints; only declared live values enter logs.
        const int64_t start =
            chunk.byte_begin / kNanCheckChunkBytes * kNanCheckChunkBytes;
        const int64_t end = std::min<int64_t>(
            hold.GetOnDeviceSizeInBytes(),
            ((chunk.byte_end + kNanCheckChunkBytes - 1) / kNanCheckChunkBytes) *
                kNanCheckChunkBytes);
        std::unique_ptr<std::byte, decltype(&std::free)> bytes(
            static_cast<std::byte*>(
                std::aligned_alloc(32, ((end - start + 31) / 32) * 32)),
            &std::free);
        if (!bytes)
          return absl::ResourceExhaustedError("FA dump allocation failed");
        auto copy =
            raiden::IssueD2hShard(hold, {{bytes.get(), start, end - start}});
        if (!copy.ok()) return copy.status();
        const auto status = copy->Await();
        if (!status.ok()) return status;
        for (int64_t offset = chunk.byte_begin; offset < chunk.byte_end;) {
          const int64_t line_begin = offset;
          std::ostringstream line;
          line << std::setprecision(std::numeric_limits<float>::max_digits10);
          for (int j = 0; j < 256 && offset < chunk.byte_end;
               ++j, offset += pool.format.element_bytes) {
            if (j) line << ',';
            line << Decode(bytes.get() + offset - start, pool.format);
            ++values;
          }
          LOG(ERROR) << "KV_NAN_DUMP_VALUES req_id=" << req << " uuid=" << uuid
                     << " pool=" << pool.index << " tag=" << pool.tag
                     << " shard=" << shard << " block_id=" << block.id
                     << " storage_byte_offset=" << line_begin
                     << " element_bytes=" << pool.format.element_bytes
                     << " values=[" << line.str() << "]";
        }
      }
  }
  LOG(ERROR) << "KV_NAN_DUMP_END req_id=" << req << " uuid=" << uuid
             << " pool=" << pool.index << " tag=" << pool.tag
             << " values=" << values;
  return absl::OkStatus();
}
}  // namespace

struct RecvNanCheck::Impl {
  // Declared first so device programs and the joined worker retire before
  // the framework's typed buffer handles are released.
  std::shared_ptr<void> inputs_owner;
  std::mutex mu;
  std::map<std::string, std::shared_ptr<Program>> programs;
  std::map<size_t, NanDumpWindow> windows;
  std::mutex work_mu;
  std::condition_variable work_ready;
  std::deque<std::function<void()>> tasks;
  bool stopping = false;
  std::thread worker;

  explicit Impl(std::shared_ptr<void> owner)
      : inputs_owner(std::move(owner)), worker([this] {
          for (;;) {
            std::function<void()> task;
            {
              std::unique_lock lock(work_mu);
              work_ready.wait(lock,
                              [this] { return stopping || !tasks.empty(); });
              if (tasks.empty()) return;
              task = std::move(tasks.front());
              tasks.pop_front();
            }
            task();
          }
        }) {}
  ~Impl() { Drain(); }
  void Schedule(std::function<void()> task) {
    {
      std::lock_guard lock(work_mu);
      assert(!stopping && "diagnostic task submitted after manager drain");
      tasks.push_back(std::move(task));
    }
    work_ready.notify_one();
  }
  void Drain() {
    {
      std::lock_guard lock(work_mu);
      stopping = true;
    }
    work_ready.notify_one();
    if (worker.joinable()) worker.join();
  }

  absl::StatusOr<std::vector<bool>> Check(
      const std::vector<ShardInput>& inputs) {
    const auto& first = *inputs[0].hold;
    const auto* api = first.c_api();
    auto handles = raiden::GetCApiDeviceHandles(first.device);
    if (!handles.ok()) return handles.status();
    PJRT_RawBuffer_GetMemorySpace_Args mem{};
    mem.struct_size = PJRT_RawBuffer_GetMemorySpace_Args_STRUCT_SIZE;
    mem.buffer = first.c_raw_buffer;
    auto status = raiden::PjrtErrorToStatusLocal(
        api, first.c_hold->extension->PJRT_RawBuffer_GetMemorySpace(&mem));
    if (!status.ok()) return status;
    std::vector<Buffer> buffers;
    std::vector<PJRT_Buffer*> args;
    std::vector<xla::Shape> shapes;
    std::vector<int64_t> metadata;
    std::ostringstream key;
    key << handles->client;
    int64_t rows = 0;
    for (const auto& input : inputs) {
      const auto& hold = *input.hold;
      const auto physical = hold.GetOnDeviceSizeInBytes();
      if (physical == 0 || physical % 4 != 0)
        return absl::InvalidArgumentError(
            "FA storage must contain whole raw words");
      const int64_t words = physical / 4;
      // Static PJRT inputs may have any shape with matching device bytes.
      // inputs_owner pins the original typed handle; this borrows it without
      // copying KV, consuming the buffer, or creating a foreign-memory view.
      args.push_back(raiden::BorrowCApiBuffer(hold.buffer));
      auto shape = xla::ShapeUtil::MakeShape(xla::U32, {words});
      shapes.push_back(shape);
      key << ':' << shape.ToString(true) << ':' << input.bucket << ':'
          << input.pool->format.element_bytes << ':'
          << input.pool->format.exponent_mask;
      for (const auto& chunk : input.chunks) {
        metadata.push_back(chunk.word_begin);
        metadata.push_back(chunk.byte_begin);
        metadata.push_back(chunk.byte_end);
      }
      metadata.resize(
          metadata.size() + (input.bucket - input.chunks.size()) * 3, 0);
      rows += input.bucket;
    }
    const std::array<int64_t, 2> dimensions = {rows, 3};
    PJRT_Client_BufferFromHostBuffer_Args meta{};
    meta.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    meta.client = handles->client;
    meta.memory = mem.memory_space;
    meta.data = metadata.data();
    meta.type = PJRT_Buffer_Type_S64;
    meta.dims = dimensions.data();
    meta.num_dims = dimensions.size();
    meta.host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    status = raiden::PjrtErrorToStatusLocal(
        api, api->PJRT_Client_BufferFromHostBuffer(&meta));
    if (!status.ok()) return status;
    buffers.emplace_back(meta.buffer, BufferDeleter{api});
    args.push_back(meta.buffer);
    if (meta.done_with_host_buffer) {
      status = raiden::PjRtCopyFuture::FromEvents(
                   api, {meta.done_with_host_buffer}, {})
                   .Await();
      if (!status.ok()) return status;
    }
    auto meta_shape = BufferShape(api, meta.buffer, xla::S64, {rows, 3});
    if (!meta_shape.ok()) return meta_shape.status();
    shapes.push_back(*meta_shape);
    std::shared_ptr<Program> program;
    {
      std::lock_guard lock(mu);
      const auto it = programs.find(key.str());
      if (it != programs.end())
        program = it->second;
      else {
        auto compiled = Compile(api, handles->client, inputs, shapes);
        if (!compiled.ok()) return compiled.status();
        program = *std::move(compiled);
        if (programs.size() >= 32) programs.erase(programs.begin());
        programs.emplace(key.str(), program);
      }
    }
    std::vector<int64_t> no_donation(args.size());
    std::iota(no_donation.begin(), no_donation.end(), 0);
    PJRT_ExecuteOptions options{};
    options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    options.non_donatable_input_indices = no_donation.data();
    options.num_non_donatable_input_indices = no_donation.size();
    PJRT_Buffer* output = nullptr;
    PJRT_Buffer* const* input_list = args.data();
    PJRT_Buffer** output_list = &output;
    PJRT_Event* complete = nullptr;
    PJRT_LoadedExecutable_Execute_Args execute{};
    execute.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute.executable = program->executable;
    execute.options = &options;
    execute.argument_lists = &input_list;
    execute.num_devices = 1;
    execute.num_args = args.size();
    execute.output_lists = &output_list;
    execute.device_complete_events = &complete;
    execute.execute_device = handles->device;
    status = raiden::PjrtErrorToStatusLocal(
        api, api->PJRT_LoadedExecutable_Execute(&execute));
    if (!status.ok()) return status;
    Buffer result(output, BufferDeleter{api});
    std::vector<uint8_t> host(inputs.size());
    const int64_t minor_to_major = 0;
    PJRT_Buffer_MemoryLayout layout{};
    layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
    layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
    layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
    layout.tiled.minor_to_major = &minor_to_major;
    layout.tiled.minor_to_major_size = 1;
    PJRT_Buffer_ToHostBuffer_Args copy{};
    copy.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    copy.src = output;
    copy.host_layout = &layout;
    copy.dst = host.data();
    copy.dst_size = host.size();
    status = raiden::PjrtErrorToStatusLocal(
        api, api->PJRT_Buffer_ToHostBuffer(&copy));
    std::vector<PJRT_Event*> events;
    if (complete) events.push_back(complete);
    if (copy.event) events.push_back(copy.event);
    auto finished =
        raiden::PjRtCopyFuture::FromEvents(api, std::move(events), {});
    const auto done = finished.Await();
    if (!status.ok()) return status;
    if (!done.ok()) return done;
    return std::vector<bool>(host.begin(), host.end());
  }
};
struct RecvNanCheck::Task::Impl {
  std::shared_ptr<RecvNanCheck> owner;
  uint64_t uuid;
  std::string req;
  std::vector<ReceivedPool> pools;
};
RecvNanCheck::RecvNanCheck(std::weak_ptr<void> inputs_owner)
    : inputs_owner_(std::move(inputs_owner)) {}
RecvNanCheck::~RecvNanCheck() = default;
void RecvNanCheck::Drain() {
  std::lock_guard lock(init_mu_);
  if (impl_) impl_->Drain();
}
RecvNanCheck::Task::Task(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
RecvNanCheck::Task::~Task() = default;
void RecvNanCheck::Task::Schedule(std::function<void()> completion) {
  impl_->owner->impl_->Schedule(std::move(completion));
}

std::unique_ptr<RecvNanCheck::Task> RecvNanCheck::CreateIfEnabled(
    kv_cache::KVCacheManagerBase* base,
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    std::shared_ptr<RecvNanCheck>* manager_owner) {
  const char* enabled = std::getenv("TPU_RAIDEN_CHECK_RECV_FA_NAN");
  if (enabled == nullptr || std::string_view(enabled) != "1") return nullptr;
  auto task = std::make_unique<Task::Impl>();
  task->uuid = plan.uuid();
  task->req = plan.req_id();
  for (const auto& group : plan.pool_groups())
    for (int32_t pool_idx : group.pool_indices()) {
      const auto& spec = *base->pool(pool_idx);
      if (!IsFullAttention(spec.tag)) continue;
      const auto format = NanFormatForDtype(spec.dtype_tag);
      auto segments = kv_cache::ExpandPoolLiveSegments(spec);
      if (!format || !segments.ok()) {
        LOG(WARNING) << "Receive FA KV NaN check skipped: req_id="
                     << plan.req_id() << " pool=" << pool_idx
                     << " tag=" << spec.tag << " dtype=" << spec.dtype_tag;
        continue;
      }
      ReceivedPool pool{
          static_cast<size_t>(pool_idx), spec.tag, *format, {}, {}};
      std::vector<NanCheckRange> ranges;
      bool aligned = true;
      for (const auto& segment : *segments) {
        ranges.push_back({segment.physical_offset, segment.size});
        aligned &= segment.physical_offset % format->element_bytes == 0 &&
                   segment.size % format->element_bytes == 0;
      }
      for (int i = 0; i < group.dst_device_block_ids_size(); ++i) {
        const int64_t block = group.dst_device_block_ids(i),
                      live = group.dst_expected_extent_bytes(i);
        aligned &= (spec.base_offset_bytes + block * spec.block_stride_bytes) %
                       format->element_bytes ==
                   0;
        aligned &= live % format->element_bytes == 0;
        pool.blocks.push_back(
            {block, BuildNanCheckChunks(spec.base_offset_bytes +
                                            block * spec.block_stride_bytes,
                                        ranges, live)});
      }
      if (!aligned) {
        LOG(WARNING) << "Receive FA KV NaN check skipped: req_id="
                     << plan.req_id() << " pool=" << pool_idx
                     << " live ranges split numeric elements";
        continue;
      }
      pool.shards = base->buffer_holds()[spec.storage_index].holds;
      if (std::any_of(pool.shards.begin(), pool.shards.end(),
                      [](const auto& h) {
                        return h.is_common_buffer || !h.shape.is_static();
                      })) {
        LOG(WARNING) << "Receive FA KV NaN check unavailable: req_id="
                     << plan.req_id() << " pool=" << pool_idx
                     << " static PJRT C API buffers required";
        continue;
      }
      task->pools.push_back(std::move(pool));
    }
  if (task->pools.empty()) return nullptr;
  if (!*manager_owner) {
    LOG(WARNING) << "Receive FA KV NaN check unavailable: req_id="
                 << plan.req_id() << " typed buffer owner not registered";
    return nullptr;
  }
  auto& diagnostic = **manager_owner;
  {
    std::lock_guard lock(diagnostic.init_mu_);
    if (!diagnostic.impl_) {
      auto inputs_owner = diagnostic.inputs_owner_.lock();
      if (!inputs_owner) {
        LOG(WARNING) << "Receive FA KV NaN check unavailable: req_id="
                     << plan.req_id() << " typed buffer owner expired";
        return nullptr;
      }
      diagnostic.impl_ = std::make_unique<Impl>(std::move(inputs_owner));
    }
  }
  task->owner = *manager_owner;
  return std::unique_ptr<Task>(new Task(std::move(task)));
}

void RecvNanCheck::Task::Run() {
  auto& owner = *impl_->owner->impl_;
  std::map<xla::PjRtDevice*, std::vector<ShardInput>> groups;
  std::map<size_t, bool> bad;
  std::map<size_t, size_t> checked;
  for (const auto& pool : impl_->pools)
    for (size_t shard = 0; shard < pool.shards.size(); ++shard) {
      ShardInput input{&pool, shard, &pool.shards[shard], {}};
      for (const auto& block : pool.blocks)
        input.chunks.insert(input.chunks.end(), block.chunks.begin(),
                            block.chunks.end());
      if (input.chunks.empty()) continue;
      while (input.bucket < static_cast<int64_t>(input.chunks.size()))
        input.bucket *= 2;
      groups[input.hold->device].push_back(std::move(input));
    }
  for (const auto& [device, inputs] : groups) {
    const auto flags = owner.Check(inputs);
    if (!flags.ok()) {
      LOG(WARNING) << "Receive FA KV NaN check unavailable: req_id="
                   << impl_->req << " uuid=" << impl_->uuid
                   << " status=" << flags.status();
      continue;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      const size_t pool = inputs[i].pool->index;
      bad[pool] = bad[pool] || (*flags)[i];
      ++checked[pool];
    }
  }
  for (const auto& pool : impl_->pools) {
    if (checked[pool.index] != pool.shards.size()) continue;
    bool dump;
    {
      std::lock_guard lock(owner.mu);
      dump = owner.windows[pool.index].OnCheck(bad[pool.index]);
    }
    if (!dump) continue;
    LOG(ERROR) << "NaN detected in received full-attention KV cache: req_id="
               << impl_->req << " uuid=" << impl_->uuid
               << " pool=" << pool.index << " tag=" << pool.tag
               << " source=device_hbm";
    const auto status = DumpPool(pool, impl_->uuid, impl_->req);
    if (!status.ok())
      LOG(WARNING) << "Receive FA KV NaN dump unavailable: req_id="
                   << impl_->req << " pool=" << pool.index
                   << " status=" << status;
  }
}
}  // namespace tpu_raiden
