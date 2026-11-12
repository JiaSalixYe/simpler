# #2855 A5 URMA 批量 PUT 性能证据

历史计时和计数使用 A5DT、CANN 9.3.0、PTO-ISA 基线 `90a3796597a5fabc2ce5066e73f785c2e85822f7`。计时使用未插桩的 ISA。各组固定每发送 rank 33 个 PUT，并校验写入数据、保护区及下游结果。

- [4 KiB 与 32 KiB 六档对照](historical-4k-32k-latency.csv)。simpler `0f6326f2b` 加 retained-domain 测试改动。计时区间从 PUT 派发到 GET 后的消费者完成。每档 80 轮，每 rank 去掉 8 轮，汇总 144 个 rank 样本，报告 p50/p95/p99 和每 rank MiB/s。
- [单向 PUT 与同窗口对照矩阵](historical-throughput-latency-matrix.csv)和[双向完整链路扫描](historical-bidirectional-sweep.csv)。simpler `b0b1eb12e`。单向的 `put` 计时到 PUT 完成后的观察任务结束，`chain` 和双向完整链路计时到 GET 后的消费者完成。矩阵按发送方向和计时终点分组，报告 p50/p95、样本数和每 rank GB/s。该数据支撑 [#2855 中的收益区间](https://github.com/hw-native-sys/simpler/issues/2855#issuecomment-6770413129)。
- [4 KiB](counts-4k.records)及[32 KiB](counts-32k.records)的逐 rank SQ doorbell 记录。计数在独立构建中运行，simpler 基线为 `0f6326f2b` 加 retained-domain 测试改动和 [simpler 探针补丁](probe-simpler.patch)。[ISA 探针](probe-isa.patch)在向实际 SQ `dbAddr` 执行 `st_dev` 后递增计数，场景使用同一 jetty 在提交调用前后的计数差。计数范围包含该 PUT 调用中的 SQ 重敲。每 rank 33 个请求，立即提交及 Batch 1、4、8、16、32 的计数分别为 33、33、9、5、3、2。CQ doorbell 和后续 GET 不在计数范围内。

## 当前 PR head f6930c89b 的同卡对照

A5DT 物理卡 1、3，CANN 9.3.0，PTO-ISA `90a3796597`。同一对卡锁定整个实验，同尺寸的两档使用相同窗口布局。双向 PUT，每个发送 rank 33 条、单 jetty。比较立即提交与选定 batch，每档 80 轮且逐轮通过数据、保护区和消费者结果校验。去掉各计时序列的前 8 轮，设备和原生 Host 运行各为每档 144 个 rank 样本，Host L3 每档为 72 个运行样本。使用最近秩法，72 个样本的 p99 等于其中最大值。

- `device_put_to_consumer` 从设备 PUT 派发到 GET 后的消费者完成。每 rank 的 33 个 PUT 有效字节除以该档平均设备时长得到 GB/s。
- `host_native_chip_run` 是每 rank 的 Host `chip.run` span，包含本次原生任务准备、观察和收尾。
- `host_l3_worker_run` 在父进程从调用 `Worker.run()` 前计到返回，含 L3 调度和通信窗口的运行内数据初始化，golden 比对在返回后执行。

下表各延迟列为立即提交 → 批量。 [逐档统计](head-summary.csv) · [逐轮样本](head-samples.csv) · [场景夹具](test_urma_poll_perf.py) · [汇总脚本](summarize.py)

| 单请求 | Batch | 设备吞吐变化 | 设备 p50（µs） | Host L3 p50（µs） | Host L3 p95（µs） | Host L3 p99（µs） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 16 | +138.3% | 222.2 → 92.8 | 10361.9 → 10760.7 | 16287.9 → 15831.8 | 25104.9 → 19620.5 |
| 32 KiB | 16 | +65.7% | 319.6 → 192.8 | 13402.2 → 16994.0 | 28630.1 → 24476.0 | 43365.3 → 29856.0 |
| 128 KiB | 4 | +15.4% | 659.7 → 571.9 | 19970.8 → 23826.1 | 29752.2 → 30433.9 | 44870.6 → 205596.7 |
| 224 KiB | 4 | -0.9% | 985.4 → 970.5 | 29471.9 → 28472.9 | 42350.0 → 38825.3 | 203129.6 → 99432.3 |
| 256 KiB | 4 | +1.1% | 1143.2 → 1140.2 | 26798.8 → 37537.1 | 48599.6 → 79704.2 | 98137.7 → 138219.4 |

4–128 KiB 的设备侧仍有合批收益，224–256 KiB 的设备吞吐差距在约 1% 内。本轮 Host L3 中位数没有随设备时长一致下降。128 KiB Batch 4 的 Host L3 p99 为 205.6 ms，对应 72 个有效样本中的最大值。这组对照比较同一提交上的立即与批量路径，没有单独改变回读实现，无法将 Host 差值归因于异步回读。224、256 KiB 为双向完整链路，历史收益反转区间来自单向 PUT，计时口径不同。

复现时，在提交 `f6930c89b` 的工作树中把夹具放到原 URMA 场景目录，用 A5 构建执行 `TestUrmaPollPerf::test_run`，逐档传入对应的 `--case`，统一使用 `--resource-child --level 3 --runtime tensormap_and_ringbuffer --manual only --platform a5 --require-pto-isa --rounds 80 --log-level timing -v -s`。用 task-submit 锁定同一对卡。夹具只为父进程 `Worker.run()` 添加计时，设备计时复用已有 STRACE。

构建产物 SHA-256：

- `aicore_kernel.o` `7c94a572f46d6bf66cab8f116cda5ee5132f1c6d477fc6efb1cbb527cddcc2f1`
- `libaicpu_kernel.so` `b71feb83d0f7660c213ce7dc00db73835274cd9cf56137cc95401750f46dd8be`
- `libhost_runtime.so` `f72a98eec584ef1db44ec7bd9f5ee54873f29178739a5a5ce7688bbf3098880f`
