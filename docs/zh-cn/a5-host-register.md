# A5 HostRegister（直接 L2）

本文描述 A5 单机分支的 Host 资源能力。该分支只实现 **A5 单进程 direct L2**；
L3 域、跨进程 Host 映射和 URMA 数据路径不在本分支范围内。

## 契约边界

- 前端复用公共 Buffer/Tensor/mapping 契约：同一 HOST Buffer 的 canonical identity
  不因映射改变；native（`mapping_ref == 0`）与 mapped view 共存。
- `map_view` 返回新的映射 Tensor；每个引用在目标 `ImportRegistry` 中登记一次
  restricted grant，记录 backing 的 target、access、extent 和 execution
  environment；A5 Worker 另外把引用绑定到发出时的 view。
- A5 后端由两部分组成：
  - **HostRegister**：`aclrtHostRegisterV2(..., ACL_HOST_REG_MAPPED)` +
    `aclrtHostGetDevicePointer`，把 POSIX shm 的 CPU backing 映射为设备地址；
  - **URMA**：仅保留 capability/failure skeleton，目前不提供数据路径；
  `Worker.host_transport_capability()` 只报告该后端是否可用。
- A5 不复制 A3 的 onboard Host owner、Fabric/VMM/NoUC 实现；两者只共享公共层。

## L2 使用方式

```python
from simpler.task_interface import (
    AccessMode, DataType, HostWindowSpec, TaskArgs, TensorArgType, TensorTransfer,
)

worker = ...  # initialized direct-L2 A5 Worker
callable = ...  # registered A5 callable
buffer = worker.create_buffer(128, host_window=HostWindowSpec(128))
native = buffer.tensor((32,), DataType.FLOAT32)

# CPU 侧可通过 buffer.shm.buf / buffer.base 初始化
mapped = worker.map_view(native, access=AccessMode.READ)

args = TaskArgs()
args.add_tensor(mapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
worker.run(callable, args)

worker.unmap_view(mapped)
worker.release_buffer(buffer)
```

`native` 只能用于 CPU 访问；A5 HostRegister 不提供 native/H2D 设备参数，设备任务参数必须是显式 mapped Tensor。
Host Buffer 由 Worker 持有，直接 `buffer.close()` 会被拒绝；必须通过
`worker.release_buffer(buffer)` 在 unmap 后释放。
A5 HostRegister 使用 POSIX shm 分配，`HostWindowSpec.numa_id` 必须为 0；
非 0 值显式拒绝，避免静默忽略 NUMA 约束。
`release_buffer` 要求原 Buffer，且所有 mapped view 已经 `unmap_view`、accepted use
已经结束；否则拒绝并保留 owner。

## 生命周期与失败语义

- 第一次 `map_view` 才执行 HostRegister；最后一次 `unmap_view` 才执行
  `aclrtHostUnregister`。CPU backing 一直保留到 `release_buffer`。
- map 失败不会留下 grant；已发生的注册会尝试立即 unregister。
- unmap/unregister 失败使记录保持 CLOSING，拒绝新的 map；重试 unmap 成功后才能
  重新映射并取得新引用。旧 Tensor 不会复活。
- 有 in-flight L2 run 或未结束 accepted use 时拒绝 `release_buffer`。
- 跨进程使用 HostRegister 记录显式拒绝；A5 单机第一刀不提供 fork 后映射语义。
- 每个 mapping_ref 在 A5 Worker 侧绑定到发出时的 native Tensor view；同一 backing
  上换用不同 offset/shape 的 view 伪造同一引用会被 `validate`/`unmap_view` 拒绝。
- runtime 未导出 HostRegister 入口（例如 A5 simulator）或 `host_register_supported`
  为 false 时，`create_buffer(host_window=...)` 显式失败，不回退到 POSIX shm
  mapping、A3 Fabric 或其他后端。

## URMA skeleton

`A5UrmaBackend` 只暴露 capability 与失败路径：

- capability 来自可选 runtime 入口；未编译/未认证时为 false；
- `register/post/wait` 直接抛出 `HostTransportUnsupported`；
- 不会因 HostRegister 可用而把 URMA 请求当作成功，也不会回退到 HostRegister。

后续 L3/URMA 实现必须复用同一公共 mapping 契约，并单独补齐 completion 和跨进程
映射验证。

## 验证口径

本分支的单元测试覆盖：native/mapped 同 identity、mapping ref 不可变且绑定 view、
mapped 参数要求 `NONE`、map/unmap 成功路径、map/unregister 失败、CLOSING 重试、
无 grant、in-flight release、继承/借用 context 拒绝、Worker.close 回收、unsupported/
a5sim、跨进程拒绝和 URMA skeleton 不回退。A5 真机
smoke 不在本分支的自动验收内；需要时在 A5 卡 0–3 上单独执行并记录环境。
