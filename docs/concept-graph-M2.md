# 概念图谱 M2 交付文档（供审阅）

> 范围：M2 = 在 M1 的融合图（`networkx.Graph`）之上做**层次社区检测**（Leiden 双分辨率）+ **防膨胀** + **受控多重归属** + **簇命名（Gemini）** + **板块归一** + **本地/GCS 持久化** + **查询接口**（`get_cluster_map`/`get_cluster_label`/`get_neighbors`），并交付可调度的离线重算脚本。
> 目标：把 M1 的「个股关系图」切分为**以概念簇为单位的分层结构**（theme 主题层 / sector 板块层），并以快照形式落盘，供方向许可层（regime gate）的 L1/L2 消费。
> 状态：已实现并用真实全量数据跑通（610 票 → 44 簇 / 12 板块）；社区检测 + 命名 + 持久化 + 板块归一 + 去重单测全部通过。

---

## 1. M2 成果一览

| 能力 | 状态 | 证据 |
|---|---|---|
| 层次 Leiden 社区检测（sector 粗 / theme 细两分辨率） | ✅ | 全量 610 票 → 44 theme 簇、12 sector |
| theme 分辨率二分搜索（簇数落入目标区间） | ✅ | `theme_target=[40,80]`，全量落在 44 |
| 防膨胀（最小簇剔除/重指派 + UNCLUSTERED） | ✅ | `min_cluster_size=4`，小簇成员按边权重指派回有效簇 |
| 受控多重归属（单票可属多个概念） | ✅ | NVDA 同时属 AI芯片(primary) + 半导体/软件(secondary) |
| 簇命名（Gemini 结构化输出，Flash-lite） | ✅ | `TH_8 → "AI Chips & Hardware"` 等语义标签 |
| **板块归一到标准 11 板块** | ✅ | `Semiconductor → Technology` 等，消除板块碎片/重复计 |
| 本地 JSON 快照 + GCS 持久化 | ✅ | `concept_graph_output/{session}/`、`gs://.../concept_graph/{session}/` |
| 查询接口（L1/L2 消费） | ✅ | `get_cluster_map`/`get_cluster_label`/`get_neighbors` |
| 可调度离线重算脚本 | ✅ | `scripts/rebuild_concept_graph.py`（盘前定时） |

---

## 2. 总体架构与数据流

```mermaid
flowchart TD
    M1[(M1 融合图\nnetworkx.Graph)] --> DET

    subgraph M2[community.py 社区检测]
        SEC[Leiden @ sector_resolution\n粗分辨率 → SEC_*]
        SRCH[二分搜索 theme 分辨率\n目标簇数区间]
        TH[Leiden @ theme_res\n细分辨率 → TH_*]
        MIN[防膨胀\n最小簇重指派/UNCLUSTERED]
        MM[受控多重归属\n边权重亲和度 → secondary]
        BUILD[_build_clusters\nparent_sector=成员众数]
    end

    DET[detect_communities] --> SEC --> SRCH --> TH --> MIN --> MM
    MIN --> BUILD
    MM --> MEMB[(memberships\nticker→[Membership])]
    BUILD --> CLU[(clusters\ncluster_id→Cluster)]

    CLU --> NAME[naming.py\nGemini 结构化命名 + 板块归一]
    NAME --> NORM[sectors.normalize_sector\n并入标准 11 板块]

    MEMB & CLU & NORM --> STORE[store.py\n本地 JSON 快照]
    STORE --> GCS[gcs.py\nGCS 持久化]
    STORE --> Q[查询接口\nget_cluster_map / label / neighbors]
    Q --> RG[regime gate L1/L2]
```

**分层原则（延续 M1）**：社区检测只消费 `networkx.Graph`，命名只消费 `Cluster`，持久化与查询统一走 `store`。换市场不影响 M2 算法；换 LLM 只换 `naming` 注入的 client。

---

## 3. 模块清单 `tradingagents/concept_graph/`

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `community.py` | 层次 Leiden + 防膨胀 + 多重归属 | `detect_communities(graph, config) -> (memberships, clusters)` |
| `naming.py` | Gemini 结构化簇命名 + 板块归一 | `name_clusters(clusters, llm=) -> clusters` |
| `sectors.py` | 标准 11 板块 + 归一映射 | `CANONICAL_SECTORS`、`normalize_sector(name)` |
| `schemas.py` | 查询接口数据契约 | `Membership`、`Cluster` |
| `config.py` | `CommunityConfig`（全部 M2 可调参数） | — |
| `store.py` | 本地 JSON 快照读写 | `save_snapshot`/`load_*` |
| `gcs.py` | 快照上传 GCS | `upload_snapshot(session, bucket, prefix)` |
| `service.py` | 端到端编排 + 查询接口 | `build_detect_save`、`name_and_save_clusters`、`get_cluster_map`/`get_cluster_label`/`get_neighbors` |

---

## 4. 原理详解

### 4.1 为什么不能用「单层硬划分」

标准 Leiden 是**硬划分**（每个节点恰好一个簇），与现实矛盾：一只票常常同时属于多个概念（NVDA 既是「AI 芯片」又是「半导体」）。M2 的做法是：**先用硬划分搭两层骨架，再在其上叠加受控的多重归属**。

### 4.2 层次 Leiden（两个分辨率）

`detect_communities`（`community.py`）在同一张图上跑两次 Leiden（`RBConfigurationVertexPartition`，按 `weight` 加权）：

1. **sector 层（粗）**：固定低分辨率 `sector_resolution=0.5` → 标签 `SEC_*`。仅用于给每个 theme 簇推断 `parent_sector`（板块）。
2. **theme 层（细）**：分辨率**二分搜索**，使簇数落入 `[theme_target_min, theme_target_max]`（市场级默认 `[40,80]`）→ 标签 `TH_*`。

**为何搜「总簇数」而非「达标簇数」**（`_search_theme_resolution`）：总簇数随分辨率**单调上升**；而「≥min_cluster_size 的簇数」非单调（分辨率过高会把大簇打碎成一堆小于阈值的碎片）。所以搜索阶段只盯总簇数（单调好二分），`min_cluster_size` 清理放到搜索之后。边界：最高分辨率仍不够 → 返回 `hi`；最低分辨率已超 → 返回 `lo`。

### 4.3 防膨胀（`_apply_min_cluster_size`）

theme 划分后，`size < min_cluster_size`（默认 4）的簇会被拆解：其每个成员按**边权重之和**重指派到「邻居所在的有效簇」中最强的一个；若该节点没有任何边连入有效簇，则标 `UNCLUSTERED`。用**原始标签**做邻居查找，保证重指派与顺序无关。

### 4.4 受控多重归属（`_multi_membership`）

每个节点：
- **primary**：theme 硬划分簇，`weight=1.0`、`is_primary=True`。
- **secondary**：对其每个邻居所属簇累加边权重，得到「亲和份额」`share = 簇内边权 / 节点总边权`；按份额降序，保留 `share ≥ multi_membership_tau`（默认 0.15）的簇，直到达到 `multi_membership_k`（默认 3，含 primary）上限。

产物 `memberships: dict[ticker, list[Membership]]`。实测：`NVDA: [(TH_8,1.0,primary), (TH_1,0.45,secondary), (TH_3,0.26,secondary)]`。

### 4.5 簇元数据（`_build_clusters`）

每个 theme 簇生成 `Cluster`：
- `parent_sector` = 簇内成员的 sector 标签**众数**（most_common）。
- `representatives` = 按加权度排序的前 `representatives_top_n`（默认 5）只票。
- `members` = 排序后的全部成员。

### 4.6 簇命名 + 板块归一（`naming.py` + `sectors.py`）

`name_clusters` 用 Gemini（默认 `gemini-3.1-flash-lite`，轻任务）结构化输出，为每个簇产出：
- `label`：具体子主题英文名（如 `AI Chips & Hardware`）。
- `parent_sector`：**从固定的 11 板块清单中选择**（prompt 内已强约束，并说明「半导体属于 Technology，勿新造板块」）。

随后对 LLM 输出再过一遍 `normalize_sector()` 兜底：把同义词/子行业并入标准 11 板块（`Semiconductor/Information Technology → Technology`、`Health Care → Healthcare`、`Materials → Basic Materials`、`Consumer Discretionary → Consumer Cyclical`、`Telecom/Media → Communication Services` 等）。未知板块**保持原样可见**（不静默合并）。

> **为什么要归一**：旧实现里 LLM 自由生成板块，导致 `Semiconductor` 与 `Technology` 并列为两个「板块」，使板块层（S3）碎片化并重复计科技敞口。归一后板块层稳定为 11 桶。同一归一函数也在 regime gate 的 `judge_sectors` 分组时调用，因此**旧快照在消费端也会被折叠**。

### 4.7 持久化与查询

- **本地快照**（`store.py`）：`{out_dir}/{session}/{edges,memberships,clusters}.json`。`edges` 用 pandas，`memberships`/`clusters` 用 Pydantic dump。命名按**交易会话日**（`label_date`），与数据日解耦（见 §5）。
- **GCS**（`gcs.py`）：`upload_snapshot(session, bucket, prefix)` → `gs://{bucket}/{prefix}/{session}/...`。
- **查询接口**（`service.py`，L1/L2 消费）：
  - `get_cluster_map(session)` → `ticker → [Membership]`（含 primary + secondary）。
  - `get_cluster_label(session, cid)` → `Cluster`（members/parent_sector/representatives/label）。
  - `get_neighbors(session, ticker, top_k)` → 按边权排序的最强邻居 `[(other, weight)]`，供 L2 催化剂传导（`propagate_catalysts`）。

---

## 5. 时间语义：会话日 vs 数据日

`rebuild_concept_graph.py` 的 `--as-of` 表示**交易会话日**（要交易的那天，盘前）。脚本：
1. `session = --as-of`（`latest` → 今天 ET）。
2. `data_date = previous_trading_day(session)`：用于建图取数的**上一交易日收盘**（价格/共现都截止到此，无未来函数）。
3. `build_detect_save(data_date, ..., label_date=session)`：用 `data_date` 的数据建图，但快照**以 `session` 命名**落盘。

因此「假设处于 6/9 盘前」时，价格/聚类用截止 6/8 收盘的数据，结果落在 `concept_graph_output/2026-06-09/`。

---

## 6. 配置参数 `CommunityConfig`

| 参数 | 默认 | 含义 |
|---|---|---|
| `seed` | 42 | Leiden 随机种子（可复现） |
| `sector_resolution` | 0.5 | 粗（板块）层分辨率 |
| `theme_resolution_lo / hi` | 0.1 / 10.0 | theme 分辨率二分搜索区间 |
| `theme_search_iters` | 18 | 二分迭代次数 |
| `theme_target_min / max` | 40 / 80 | 目标 theme 簇数区间（市场级） |
| `min_cluster_size` | 4 | 小于此的 theme 簇被重指派/标 UNCLUSTERED |
| `multi_membership_tau` | 0.15 | 添加 secondary 归属的最小亲和份额 |
| `multi_membership_k` | 3 | 每票最多归属簇数（含 primary） |
| `representatives_top_n` | 5 | 每簇保留的代表票数（按加权度） |

---

## 7. 测试与实测证据

**单元测试（合成图 / mock LLM，`pytest`）**：
- `test_community.py`：planted 簇结构恢复、最小簇重指派、多重归属阈值、命名填充 label/sector（含 `Semiconductor → Technology` 归一断言）、GCS 上传。
- `test_sectors_tickers.py`：`normalize_sector` 折叠子行业/同义词、大小写不敏感、未知保留可见。

**真实数据冒烟（全量 610 票）**：
- 44 theme 簇 / 610 个成员关系 / 12 板块；每簇成员 min/中位/max = 4/12/40。
- 标签语义合理：`Big Tech & E-commerce`、`Mining & Metals`、`Alternative Asset Managers`、`Oil, Gas & Energy Infrastructure` 等。
- 多重归属生效：`NVDA`、`AMD`、`WMT` 等出现合理的 secondary 归属。

**过程中修复的真实 bug（fail-fast）**：
- `_search_theme_resolution` 早期用「达标簇数」二分（非单调）导致 planted 测试 `assert 1 == 3` → 改为搜总簇数 + 事后 `min_cluster_size` 清理。
- `latest_trading_day`/分区表必须带 `trade_date` 过滤 → 加 lookback 窗口避免 BigQuery 全表扫描报错。

---

## 8. 已知边界与后续

- **`fuse_gamma`（ETF 共同归属边）** 仍占位未启用（受历史成分数据限制）。
- **跨日簇稳定性**：当前每日独立重算，簇 id（`TH_*`）不保证跨日一致；如需追踪同一主题的时间演化，需加 id 对齐/匹配层。
- **板块归一表** 为人工维护的同义词映射，新出现的板块别名需补充进 `sectors._ALIASES`。

---

## 9. 用法

```python
from tradingagents.dataflows.secrets import load_secrets_to_env
from tradingagents.concept_graph.service import build_detect_save, name_and_save_clusters, get_cluster_map

load_secrets_to_env()

# 全量：建图 + 社区检测 + 落盘（数据用 data_date，快照命名用 session）
edges, g, memberships, clusters = build_detect_save(
    "2026-06-08", label_date="2026-06-09",     # data_date / session
)
# 命名 + 板块归一 + 重存
clusters = name_and_save_clusters("2026-06-09")

# 消费（regime gate L1/L2）
cmap = get_cluster_map("2026-06-09")           # ticker -> [Membership]
```

命令行（盘前定时）：

```bash
# 全量 + 命名 + 上传 GCS
python scripts/rebuild_concept_graph.py --as-of latest --all --name \
    --gcs-bucket trading_agent --gcs-prefix concept_graph
```
