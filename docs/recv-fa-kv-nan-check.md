# 接收 FA / GDN KV 的异步 TPU NaN/Inf 诊断

## TL;DR

`TPU_RAIDEN_CHECK_RECV_FA_NAN=1` 开启接收端诊断：pool-reshard receive 的全部 H2D 完成后，使用 native PJRT 在 TPU 上检查本次接收的 FA、GDN conv/SSM 有效范围，只回读每个 pool/shard 的布尔结果。所有设备的检查和异步回读提交后即报告接收完成；不等待检查执行、bool 到达 host 或日志输出。检查必须先于本次 forward 提交，并依赖接收进程的设备同流执行顺序。默认关闭，不进入诊断路径。

每个 manager 的每层/pool 按每 100 次成功完成的接收检查划分窗口，包括正常结果。每个窗口仅前 3 次 NaN/Inf 命中输出异常 tensor 的定位日志；其余命中只做 TPU 检查和布尔回读。此版本不回读或输出 KV 数值，不创建 KV 快照。第 101、201……次检查开始新窗口；重建 manager 后计数重置。

## 使用和检查范围

在 **decode/接收进程** 设置：

```bash
export TPU_RAIDEN_CHECK_RECV_FA_NAN=1
```

只有值恰好为 `1` 才开启。发送端、普通非 pool-reshard receive 不受影响。每次创建接收 session 时读取开关。

检查 tag 为 `fa`、`fa.l<十进制层号>`、`gdn.conv` 或 `gdn.ssm` 的 pool；GDN 还接受 `.g<十进制组号>` 和 `.l<十进制层号>` 后缀，组号在层号之前。dtype 必须明确是 `bfloat16`/`bf16`、`float16`/`f16` 或 `float32`/`f32`。FP8、整数 carrier、未知 dtype 跳过，不按字节猜测格式。GDN 必须使用逻辑 conv/SSM pool 的真实 dtype，而非承载整页的 raw Tensor dtype。BF16、FP32 可共用同一 backing allocation。正负 Inf 和 NaN 均视为异常。

使用 plan 中该 pool 的目的 block IDs 和每个 block 的 `dst_expected_extent_bytes`。extent 表示各 live region 拼接后的有效前缀，因此 region 间的空洞、没有传输的 block、有效前缀后的尾部均不参与检查，也不参与结果归约。数值范围必须按元素宽度对齐。检查对象是 **所有 pool H2D 完成后的 device HBM**；如果不同 pool 的物理地址别名，看到的是最终 HBM 内容。

## 执行与生命周期

manager 惰性持有一条独立的 native 诊断线程、最多 32 个编译程序和每个 pool 的限频状态；不调用 Python/JAX 或 `.item()`。全部 H2D 成功后，线程提交各设备的检查及异步 bool 回读，再通知 session 完成接收。仅排入 host 任务队列不算完成提交。

线程以 1 ms 等待间隔轮询 pending 完成事件，事件全部完成且状态成功后才读取 host `std::vector<uint8_t>`：每个逻辑 pool/shard 一个字节，0 表示无 NaN/Inf，1 表示存在 NaN/Inf。tensor 的 tag、dtype、pool/shard 对应关系保存在 host。结果按提交顺序消费，保持每个 pool 的 100 次检查窗口顺序；一次 pool 的检查只有全部 shard 成功才计数，诊断失败不当作无 NaN/Inf。

pending task 独立持有 typed buffer owner、范围元数据、程序、结果 buffer、host 目标和 PJRT 事件，不依赖接收 session 或 staging arena。manager 析构先结算 session，再 drain 诊断线程，最后释放设备资源。正常提交/轮询不调用 `Await`；异常展开时尚未结束的设备操作会在资源销毁前 drain。取消不提前释放已提交检查的资源。

最多接纳 128 个待提交/待回读检查，满载或 drain 后不再接纳新诊断，记录检查跳过原因并让传输继续；跳过不计入成功检查窗口。诊断仅用于异步观测，结果不回写已发布的接收状态。

TPU 程序按原始 U32 字节布局进行位分类及布尔归约，直接借用已注册的 PJRT buffer，不复制 KV、不创建额外的 device-memory view、不消耗原 buffer。依赖 PJRT 的 `strict_shape_checking=false` 静态输入契约：输入原 shape 可以不同，但设备字节数必须与编译形状一致。范围元数据上传到设备；不同范围数量按 2 的幂分桶复用编译程序。第一次遇到新 geometry 有编译开销，后续命中缓存。所有输入禁止 donation。原 buffer 的 PJRT usage tracker 管理设备使用期间的底层 allocation。

同一 storage/shard 只传入一个原始 buffer 参数；共享该 buffer 且元素格式、chunk bucket 相同的逻辑 pool 合并为一个批量扫描图，仍分别归约每个 pool 的结果。每个 chunk 仍扫描 32 KiB，每个 pool 的有效字节范围、空洞和 bucket padding 独立 mask；不减少检查范围或元数据行数。扫描组按 storage、shard、元素宽度、exponent mask 和 bucket 排序，plan 中 pool 顺序改变仅调整元数据及结果映射，不产生新的程序签名。程序缓存仍归属于 manager，key 包含 client、去重后的 U32 shapes、扫描组 geometry 和元数据 layout；新 bucket 和新 geometry 仍可能同步编译。

每次真实编译输出 `Receive KV diagnostic compile`，记录 key hash、完整签名、cache size/eviction、逻辑 pool/storage/扫描组数量、metadata rows 和编译耗时。`VLOG(1)` 另外记录 queue、metadata 准备/上传和 layout、Execute/回读提交分段耗时；这些是 host 提交耗时，不等同于 TPU 设备执行时间。编译及全部设备提交仍先于接收完成回调。

Torch 在构造 manager 时为已有 `TensorBufferHandle` owner 注册弱引用。首次开启且存在可检查 pool 时，诊断对象提升为强引用，覆盖后续任务、设备操作和执行器 Drain；关闭时不创建诊断线程或编译程序，原 buffer 的释放时点保持。每个 session 读取开关，支持先关闭再开启。要求静态 PJRT C API buffer 和已注册的 typed buffer owner；raw-pointer-only frontend、Common client 和动态 shape 不进入设备诊断。不支持配置或执行错误记录诊断不可用，传输本身不因诊断失败而报错，也不退回 CPU 全量扫描。

## 异常 tensor 定位日志

仅对窗口内获准的 NaN/Inf 命中输出一条 pool 定位日志，例如：

```text
NaN/Inf detected in received KV cache: req_id=... uuid=... pool=... tag=gdn.ssm.g0.l12 dtype=float32 shards=[0,2] source=device_hbm
```

`shards` 只列实际命中的 shard；tag 标识 FA 或 GDN conv/SSM 及其组/层。任一 shard 命中只占一次该 pool 的日志额度。正常结果和超出额度的 NaN/Inf 不输出此日志。GDN conv 与 SSM 是独立逻辑 pool，各自计数。

这是本次有效接收范围的 bool 结果，不包含首个 NaN/Inf 偏移、具体异常 block 或 KV 数值。不在 bool 到达后重读原 HBM，因此即使 forward 已更新 GDN state，定位结果仍对应此前提交的接收检查。关闭功能时不创建线程、程序或 pending 资源。
