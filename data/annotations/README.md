# Annotations

将与 manifest 一一对应的 `parking_risk_v1.jsonl` 人工标注放在本目录。冻结测试集后，
LoRA 调参、量化校准和阈值调整不得改写该文件。

严格输出字段和取值约束见 [`docs/data.md`](../../docs/data.md)。

标注来源必须在实验摘要和生成数据时显式记录。`ps80_reviewed_v1` 当前是
`codex_visual_review_v1_single_pass` 候选标注，不是人工金标；人工终审后应使用独立文件名
和来源标识，例如 `ps80_human_confirmed_v1` / `human_confirmed_v1`，再进入 LoRA 或 INT4
数据生成流程。
