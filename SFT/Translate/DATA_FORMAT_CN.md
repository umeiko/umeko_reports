# 数据格式、划分和生成规范

## 1. 清洗后的唯一句对

推荐保存为 Parquet；下面用 JSON 展示字段：

```json
{
  "pair_id": "sha256:...",
  "source_dataset": "opus100",
  "source_revision": "commit-or-version",
  "source_row_id": "train:12345",
  "document_id": null,
  "segment_index": null,
  "english": "The meeting starts at nine.",
  "chinese_simplified": "会议九点开始。",
  "license_id": "VERIFY_OR_SPDX",
  "lid_en": 0.99,
  "lid_zh": 0.99,
  "alignment_score": 0.91,
  "quality_flags": [],
  "split": "train"
}
```

`pair_id` 基于规范化后的两端内容、来源和原始文档标识生成。原文不能因去重规范化而被覆盖。

## 2. 方向化后的 SFT 样本

```json
{
  "example_id": "<pair_id>:en-zh-Hans:template_v1",
  "pair_id": "sha256:...",
  "direction": "en-zh-Hans",
  "source_locale": "en",
  "target_locale": "zh-Hans",
  "prompt": "Translate the following text from English to Simplified Chinese.\nEnglish: The meeting starts at nine.\nSimplified Chinese:",
  "target": "会议九点开始。",
  "template_id": "en_zh_v1",
  "split": "train"
}
```

反向样本复用同一个 `pair_id`。必须先按 pair/document 划分，再方向化；禁止先翻倍后随机切分。

## 3. loss mask

拼接 token 后：

```text
input_ids = prompt_ids + target_ids + [eos]
labels    = [-100] * len(prompt_ids) + target_ids + [eos]
```

若 packing 多条样本，除了 label mask，还必须阻断跨样本注意力或使用训练框架验证过的 sequence boundary 实现。

## 4. 文档样本

只有原始数据给出真实 `document_id` 和连续 `segment_index` 时，才能构造上下文：

```json
{
  "context_before": ["上一句……"],
  "source": "当前句……",
  "target": "Current sentence...",
  "document_id": "doc-42",
  "segment_index": 7
}
```

随机把三句无关文本拼在一起，不能训练或评价“文档能力”。

## 5. manifest 最低字段

每个数据文件都生成 manifest，至少包含：

- 下载 URL、获取日期、版本/revision、原始 SHA-256。
- 原始条数、逐步骤删除条数、最终条数。
- 语言识别器和阈值、对齐模型和阈值、去重算法版本。
- 与每个评测集的 exact/near-overlap 数量和处置。
- 许可证标识、审核人、是否允许发布派生物。
- train/dev/test 的 pair 和 document 数量。

没有 manifest 的数据不得进入正式训练。

