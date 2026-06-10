# 概念图谱全量运行结果分析（2026-06-08 快照）

> 数据：全候选池（`valid_ticker_v3_pure_cs`，约 610 只 CS）于 `2026-06-08` 的快照。
> 参数：`GraphConfig`（comovement_window=120, prune_top_k 默认）+ `CommunityConfig` 默认（theme_target 40–80, min_cluster_size=4, 多归属 τ=0.15/K=3）。
> 产物：`concept_graph_output/2026-06-08/{edges,memberships,clusters}.json` + GCS `gs://trading_agent/concept_graph/2026-06-08/`。

---

## 1. 总览

- **44 个 theme 簇**（TH_0–TH_43），平均约 14 只/簇，落在默认目标区间 40–80（指总簇数搜索目标）内，粒度适中。
- 簇语义整体清晰，板块/主题分层正确；`parent_sector` 多数准确（少量 LLM 主观偏差）。
- 结论：**结果可直接供主方案 L2 概念聚合消费**，代码无需修改。

---

## 2. 高质量簇（亮点）

| 簇 | label | 评价 |
|---|---|---|
| TH_7 | Asset Management & Alternative Investments | KKR/BX/APO/ARES/OWL/BDC 等，私募/另类资管独立成簇，专业 |
| TH_23 | Midstream Energy & Pipelines | 中游管道（OKE/EPD/WMB/ET/KMI）与上游油气（TH_1）正确分离 |
| TH_1 | Oil, Gas & Energy Infrastructure | 上游+油服干净聚类 |
| TH_4 | Mining & Precious Metals | 金银矿（NEM/AG/KGC/FCX）精准 |
| TH_11 | Data Centers & Semiconductor Infra | NVDA/AVGO/CRWV/CORZ + 加密矿工 RIOT/HUT/WULF 因算力 co-move，有洞察 |
| TH_0 | Semiconductor Equipment & Industrial Automation | 半导体设备/材料聚类 |
| TH_6 | Enterprise SaaS & Cloud Infrastructure | 云软件/安全（CRM/NOW/SNOW/PANW/CRWD） |
| TH_41 | Latin American Fintech | NU/STNE/PAGS/XP/JBS，地域+赛道双精准 |
| TH_43 | Solar Energy | ENPH/RUN/SEDG/ARRY |
| TH_12 | Disruptive Innovation & Digital Assets | COIN/MSTR/HOOD/PLTR/TSLA/SOFI，高 beta 成长+加密篮子 |
| TH_15 / TH_25 | 权益型 REITs / 抵押型 REITs | 两类 REIT 正确分层 |
| TH_2 / TH_5 / TH_21 | 银行 / 公用事业+烟草 / 医疗器械 | 经典板块到位 |

## 3. 边缘归类（单日 co-movement 噪声，非缺陷）

1. **AAPL 跑偏**：归入 `TH_20 Travel & Transportation`（航空+邮轮）。AAPL 在该 120 日窗口的去市场化残差与主流科技簇不够强，被边缘吸附（小样本时亦如此）。
2. **TH_13 防御混合**：大盘防御股（LLY/JNJ/MRK + MCD/WMT/SYY + RTX）抱团，是"防御因子"co-move，语义偏杂。
3. **TH_3 题材篮子**：铀矿/小堆(OKLO/SMR/UEC) + 量子(IONQ/RGTI/QBTS) + 太空(RKLB/ASTS)——真实的散户题材高相关篮子，主题偏宽但可接受。
4. **TH_17**：biotech(MRNA/NTLA) 与 EV(LCID/RIVN) 并簇，高波动成长股 co-move。
5. **parent_sector 偶有 LLM 偏差**：如 TH_42 含 REIT(AMT/CCI) 却标 Consumer Cyclical。

## 4. 根因与改进方向

- 边缘漂移的根因是**单日 co-movement 受防御/题材因子主导**，而非算法缺陷。
- **首选改进：M3 跨日稳定性对齐**（连续 N 日成员重叠率过滤 + 簇 id 继承），系统性消除单日抖动，优于调参。
- 可选即时微调（仅改 `GraphConfig`，不改代码）：
  - `fuse_alpha` 0.4→0.5：抬高新闻共现权重，语义更主导，AAPL 类更易回归科技；
  - `comovement_window` 120→180：相关估计更稳，降低单日噪声。
- `parent_sector` 偏差可在 `naming.py` prompt 内给固定 GICS 板块词表约束（低优先）。

## 5. 运维确认

- `--as-of latest` 已修复分区过滤问题（`day_aggs_di` 需 `trade_date` 过滤），回看 14 天取最近交易日。
- 每日盘前经 `scripts/run_concept_graph_daily.sh`（美东工作日闸门）+ cron（北京 20:00）运行，结果写本地并上传 `gs://trading_agent/concept_graph/<date>/`。
