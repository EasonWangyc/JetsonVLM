# 数据约定

`data/manifests/` 与 `data/annotations/` 使用 UTF-8 JSONL。每一行对应一条记录，空行会被忽略。

## Manifest

manifest 记录必须且只能包含以下字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `case_id` | 非空字符串 | 泊车样本的稳定标识 |
| `image_ref` | 相对 POSIX 路径 | 相对数据根目录的图片引用，例如 `raw/case-001.jpg` |
| `source_group_id` | 非空字符串 | 原始视频、连续采集序列或不可分数据来源的标识 |
| `split` | `train`、`validation` 或 `test` | 样本的数据集划分 |

`image_ref` 不得是绝对路径、不得包含 `..`，也不得使用 Windows 反斜杠。相同的 `source_group_id` 必须只出现于一个 `split`，避免同一来源泄漏到训练、验证或冻结测试集。

```json
{"case_id":"case-001","image_ref":"raw/case-001.jpg","source_group_id":"sequence-a","split":"test"}
```

## Annotation

annotation 记录必须且只能包含 `case_id` 与 `assessment`。`case_id` 必须与 manifest 一一对应；`assessment` 遵循 `ParkingAssessment` 的严格 JSON 契约。

```json
{"case_id":"case-001","assessment":{"schema_version":"parking_risk_v1","risk_level":"medium","events":["narrow_passage"],"evidence":["Vehicles leave a narrow maneuvering corridor."],"driver_advice":["slow_down"]}}
```
