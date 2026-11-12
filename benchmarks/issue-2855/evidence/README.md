# #2855 A5 URMA 批量 PUT 性能证据

历史计时和计数使用 A5DT、CANN 9.3.0、PTO-ISA 基线 `90a3796597a5fabc2ce5066e73f785c2e85822f7`。计时使用未插桩的 ISA。各组固定每发送 rank 33 个 PUT，并校验写入数据、保护区及下游结果。

- [4 KiB 与 32 KiB 六档对照](historical-4k-32k-latency.csv)。simpler `0f6326f2b` 加 retained-domain 测试改动。计时区间从 PUT 派发到 GET 后的消费者完成。每档 80 轮，每 rank 去掉 8 轮，汇总 144 个 rank 样本，报告 p50/p95/p99 和每 rank MiB/s。
- [单向 PUT 与同窗口对照矩阵](historical-throughput-latency-matrix.csv)和[双向完整链路扫描](historical-bidirectional-sweep.csv)。simpler `b0b1eb12e`。单向的 `put` 计时到 PUT 完成后的观察任务结束，`chain` 和双向完整链路计时到 GET 后的消费者完成。矩阵按发送方向和计时终点分组，报告 p50/p95、样本数和每 rank GB/s。该数据支撑 [#2855 中的收益区间](https://github.com/hw-native-sys/simpler/issues/2855#issuecomment-6770413129)。
- [4 KiB](counts-4k.records)及[32 KiB](counts-32k.records)的逐 rank SQ doorbell 记录。计数在独立构建中运行，simpler 基线为 `0f6326f2b` 加 retained-domain 测试改动和 [simpler 探针补丁](probe-simpler.patch)。[ISA 探针](probe-isa.patch)在向实际 SQ `dbAddr` 执行 `st_dev` 后递增计数，场景使用同一 jetty 在提交调用前后的计数差。计数范围包含该 PUT 调用中的 SQ 重敲。每 rank 33 个请求，立即提交及 Batch 1、4、8、16、32 的计数分别为 33、33、9、5、3、2。CQ doorbell 和后续 GET 不在计数范围内。
