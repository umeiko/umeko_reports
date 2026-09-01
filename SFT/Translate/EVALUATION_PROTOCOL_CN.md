# 中英翻译冻结评测协议

本文件的目的，是防止看到测试分数后不断改提示词和超参数。任何正式结果必须记录本文件版本。

## 1. 数据角色

| 角色 | 数据 | 允许做什么 | 禁止做什么 |
|---|---|---|---|
| train | 清洗后的训练池 | 更新参数 | 与评测数据重叠 |
| internal dev | 固定 10K 唯一句对，双向 | 看曲线、早停、选 checkpoint | 写进最终排行榜主表冒充外部测试 |
| public dev | FLORES+ dev | 所有设置初步冻结后做一次 sanity check | 反复调 prompt、beam、学习率 |
| final test | FLORES+ devtest；WMT22–26；WMT24++；NTREX；TICO test | 只在全部设置冻结后生成与打分 | 用于训练、选 checkpoint 或调解码 |

WMT20/21 可任选一个作为历史解码开发集，但必须预先选定；不能在多个年份上试完再挑最好配置。WMT17–19 仅作传统可比性附录，不承担主结论。

## 2. 固定的输入输出

主模型只翻译简体中文 `zh-Hans` 和英文 `en`。WMT24++ 使用 `en-zh_CN`。繁体中文另设 `zh-Hant-TW` 配置，未完成专门训练前不混入简体主结果。

评测提示固定为：

```text
Translate the following text from English to Simplified Chinese.
English: {source}
Simplified Chinese:
```

```text
请将以下简体中文翻译成英文。
简体中文：{source}
英文：
```

只抽取提示词之后、首个 EOS 之前的文本作为译文。不得针对某个测试集手工改提示。

## 3. 固定的解码

- 主结果：greedy，`do_sample=false`，temperature/top-p 不生效。
- beam 消融：`num_beams=4`，`length_penalty=1.0`，只运行一次预注册比较。
- 不用候选重排，不用外部 LLM 后编辑。
- 设方向相关的安全长度上限；任何达到上限的样本记为 truncated，不能静默截断。
- 文档数据保持原始顺序。若模型上下文不足，采用预先定义的滑窗，不得按答案长度手工切分。

## 4. 自动指标

每个数据集、方向和领域分别报告：

1. SacreBLEU BLEU，连同完整 version signature。
2. chrF++，word order 为 2。
3. COMET：固定 `Unbabel/wmt22-comet-da` 的精确 revision。
4. 可选 MetricX-24-Hybrid-Large/XL。注意它输出的是误差分，越低通常越好。
5. 输出审计：目标语言正确率、空输出、源文复制、截断、数字/实体保持、结构化格式合法率。

中文目标的 BLEU tokenizer 使用测试集官方约定；没有特殊要求时使用 SacreBLEU `zh`。英文目标通常为 `13a`。绝不把不同 tokenizer、不同参考版本的 BLEU 放进同一列比较。

## 5. 统计比较

- 用相同样本上的 paired bootstrap resampling，1,000 次，报告 95% CI。
- 关键结论使用 paired approximate randomization，10,000 次。
- 神经指标比较可使用 COMET 的 `comet-compare`。
- 1M 主设置跑 3 个种子，报告均值和标准差；单次最好结果只能作补充。
- 不设一个“万能平均分”决定胜负。双方向和不同领域必须可见。

## 6. 人工评价

每方向随机分层抽 300–500 段；文档测试按文档抽样而非打散。至少 20% 由两位标注者独立评价，分歧仲裁。标注者不知道系统名称。

记录错误 span、类别和严重度：

- 准确性：误译、漏译、增译、未翻译。
- 流畅性：语法、搭配、标点、重复。
- 术语和一致性。
- 专名、数字、日期、单位。
- locale：简繁体和地区表达。
- 指令/格式：术语表、JSON/HTML/CSV 合法性和风格要求。

主报告给出每千词错误数、重大错误数和置信区间，同时给出代表性成功/失败案例。不要只展示好看的例子。

## 7. WMT 年份与方向说明

- SacreBLEU 当前可直接获取的 WMT17–23 包含英中与中英方向。
- WMT24、WMT25、WMT26 的主方案里重点报告英→简中；以每届官方发布的实际方向和参考为准。
- WMT26 还包含英→台湾繁中，但它不是简中结果，必须分栏。
- WMT26 正式提交已于 2026-07-02 截止；当前运行标记为 `post-hoc/offline`，不能写成官方排名。
- WMT27 尚未公布时，不预设日期、语言方向或 track。公告后新增版本化附录，不覆写本协议的历史结果。

## 8. 结果表模板

| 模型 | 参数 | 训练唯一对 | 方向 | 数据集 | BLEU↑ | chrF++↑ | COMET↑ | 语言正确率↑ | 截断率↓ | 95% CI | 运行 ID |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| Haidass base | 143M | 0 | en→zh-Hans | FLORES+ devtest | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待算 | base-eval |
| Translate-SFT | 143M | 1M | en→zh-Hans | FLORES+ devtest | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待算 | a1-seedX |

## 9. 每次评测的强制产物

- `sources.jsonl`、`references.jsonl`、`hypotheses.jsonl`，行数与 ID 对齐。
- `run_manifest.json`：模型、checkpoint、数据、prompt、解码和环境哈希。
- `metrics.json` 和人可读 Markdown 表。
- SacreBLEU signature、COMET/MetricX model revision。
- 逐句指标和错误审计，供 paired 统计与复核。
- 任何失败、重跑和参数变化的日志。

