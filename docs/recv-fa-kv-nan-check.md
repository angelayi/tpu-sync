# 接收 FA KV 的 TPU NaN 诊断

## TL;DR

`TPU_RAIDEN_CHECK_RECV_FA_NAN=1` 开启接收端诊断：pool-reshard receive 的全部 H2D 完成后，使用 native PJRT 在 TPU 上检查本次接收的 FA KV 有效范围，只回读每个 pool/shard 的布尔结果。接收完成事件等待检查和布尔回读；允许 dump 的 NaN 命中还会等待 KV 回读与日志输出。默认关闭，不进入诊断路径。

每个 manager 的每层/pool 按每 100 次成功完成的接收检查划分窗口，包括正常结果。每个窗口仅前 3 次 NaN 命中输出 NaN 日志及该层本次接收的 KV；其余命中只做 TPU 检查和布尔回读，不回读 KV、不生成 KV 日志。第 101、201……次检查开始新窗口；重建 manager 后计数重置。

## 使用和检查范围

在 **decode/接收进程** 设置：

```bash
export TPU_RAIDEN_CHECK_RECV_FA_NAN=1
```

只有值恰好为 `1` 才开启。发送端、普通非 pool-reshard receive 不受影响。每次创建接收 session 时读取开关。

检查 tag 为 `fa` 或 `fa.l<十进制层号>` 的 pool，且 dtype 必须明确是 `bfloat16`/`bf16`、`float16`/`f16` 或 `float32`/`f32`。FP8、整数 carrier、未知 dtype 跳过并给出说明，不按字节猜测格式。正负 Inf 不算 NaN。

使用 plan 中该 pool 的目的 block IDs 和每个 block 的 `dst_expected_extent_bytes`。extent 表示各 live region 拼接后的有效前缀，因此 region 间的空洞、没有传输的 block、有效前缀后的尾部均不参与检查，也不会进入 dump。数值范围必须按元素宽度对齐。检查对象是 **所有 pool H2D 完成后的 device HBM**；如果不同 pool 的物理地址别名，看到的是最终 HBM 内容。

## 执行与生命周期

manager 惰性持有独立的 native 诊断线程、最多 32 个编译程序和每个 pool 的限频状态；不调用 Python/JAX。session 固定本次计划的范围与 device buffer 引用，并在全部 H2D 成功后将任务排入该 manager 的诊断队列。任务保留 receive operation，覆盖 TPU 执行、布尔回读和获准的 dump；在结束前不发布接收完成。取消、超时和 manager 析构仍等待已开始的设备操作结束，退出诊断线程后再释放资源。

TPU 程序按原始 U32 字节布局进行位分类及布尔归约，直接借用已注册的 PJRT buffer，不复制 KV、不创建额外的 device-memory view、不消耗原 buffer。依赖 PJRT 的 `strict_shape_checking=false` 静态输入契约：输入原 shape 可以不同，但设备字节数必须与编译形状一致。范围元数据上传到设备；不同范围数量按 2 的幂分桶复用编译程序。第一次遇到新 geometry 有编译开销，后续命中缓存。所有输入禁止 donation。原 buffer 的 PJRT usage tracker 管理设备使用期间的底层 allocation。

Torch 在构造 manager 时为已有 `TensorBufferHandle` owner 注册弱引用。首次开启且存在可检查 pool 时，诊断对象提升为强引用，覆盖后续任务、设备操作和执行器 Drain；关闭时不创建诊断线程或编译程序，原 buffer 的释放时点保持。每个 session 读取开关，支持先关闭再开启。要求静态 PJRT C API buffer 和已注册的 typed buffer owner；raw-pointer-only frontend、Common client 和动态 shape 不进入设备诊断。不支持配置或执行错误记录诊断不可用，传输本身不因诊断失败而报错，也不退回 CPU 全量扫描。

## 日志与 KV dump

获准的 NaN 命中输出：

```text
NaN detected in received full-attention KV cache: req_id=... uuid=... pool=... tag=fa.l... source=device_hbm
KV_NAN_DUMP_BEGIN req_id=... uuid=... pool=... tag=... source=device_hbm
KV_NAN_DUMP_VALUES req_id=... uuid=... pool=... tag=... shard=... block_id=... storage_byte_offset=... element_bytes=... values=[...]
KV_NAN_DUMP_END req_id=... uuid=... pool=... tag=... values=...
```

这是 pool 级布尔诊断，不报告首个 NaN 偏移。`VALUES` 行中的 offset 是该行首元素相对 storage allocation 的物理字节偏移。输出该层本次接收的所有有效 KV 数值，包含所有 shard；每行最多 256 个元素，不截断。只要该 pool 任一 shard 命中 NaN，就占用一次该 pool 的 dump 配额。

KV 回读使用独立临时内存，不覆盖网络/H2D staging arena。限频在 KV 回读和格式化之前决定。每百次检查窗口中获准的三次完整 dump 可能产生大量日志与额外延迟；该开关用于排查，不是默认服务配置。
