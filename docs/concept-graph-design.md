# 个股相关性知识图谱（概念集涌现 + 催化剂传导）子方案

> 版本：v0.1（草案）
> 关系：本子系统是主方案 `regime-gate-design.md` 中 **P3「概念集映射」** 的实现，可独立开发与部署。
> 一句话定位：用「共现 + 共振」两类信号构建个股相关性图，通过社区发现**动态涌现概念集**（替代手工维护的静态板块表），并支持**催化剂在相关个股间的图传导**。

---

## 1. 目标与定位

主方案 L2 概念层原本需要一张静态的 `ticker → 板块（芯片/存储…）` 映射表。问题是：静态表维护成本高、滞后、且无法表达「跨板块的隐性联动」（如某存储股因 AI 需求与芯片股强联动）。

本子系统改为**从数据涌现概念集**：

- **节点（Node）**：仅个股（Common Stock）。
- **边（Edge）**：个股间的相关性强度，由两类信号融合：
  1. **Co-mention（共现）**：两 ticker 在同一篇新闻/财报中被同步提及。
  2. **Co-movement（共振）**：两 ticker 价格同步涨/跌（去市场化后的收益率相关性）。
- **ETF 处理**：ETF **不作为节点**，而是展开为其成分股——既用于把 ETF 相关新闻的共现信用分摊给成分，也作为一类「共同归属」先验边。

**两大产出**：
1. `ticker → cluster_id` 动态概念集映射（+ 簇标签），供主方案 L2 直接消费。
2. 相关性边表，支持 L1 的**催化剂图传导**（一只股票出强催化剂，按边权重提升其强相关邻居的置信度）。

---

## 2. 设计原则

- **个股为本**：图中只有个股节点；指数/ETF 是「视角」不是「实体」。
- **去市场化**：co-movement 必须剔除大盘共同涨跌的伪相关，否则牛市里所有股票都高相关，社区检测失效。
- **归一化**：co-mention 必须消除「大票到处被提及」的偏置（用 PMI / Jaccard，而非原始计数）。
- **时间衰减**：近期信号权重更高。
- **Fail-fast**：数据不足（样本量低于阈值）的边直接不建立，不做可疑插值。
- **批处理 + 可重算**：图按日/周离线重算，结果落库；不引入在线图数据库的运维负担（除非规模要求）。
- **多重归属**：单个 ticker 可同时属于多个细分主题簇（如 NVDA ∈ {AI 推理芯片, 数据中心, HBM 产业链}），用「主归属 + 带权次归属」表达。
- **受控粒度**：追求细分主题，但用最小簇规模、簇数上限、层次截断、稳定性过滤四重机制防止簇无限膨胀。

---

## 3. 数据来源

| 信号 | 来源（已落定） | 字段 |
|---|---|---|
| Co-mention | Massive `/v2/reference/news`、Benzinga `/benzinga/v2/news`、FMP `/stable/news/*` | `results[].tickers` / `stocks` 数组 |
| Co-movement | **BQ 日线** `day_aggs_di`（Polygon/Massive 风格，非复权，≥2024） | `ticker, open, high, low, close, volume, transactions, trade_date(DATE)` → 日收益率 |
| 拆股修正 | **Massive `/v3/reference/splits`**（历史，单票一次调用，量小，缓存到 BQ） | `execution_date, split_from, split_to` |
| 去市场化基准 | `macro_daily.nq_futures_close` / `nq_futures_pct_change`（亦可从日线表自算 QQQ/SPY 代理） | 市场因子收益率 |
| 盘中增量 | **BQ 分钟** `minute_aggs_di` | `ticker, open, high, low, close, volume, transactions, trade_minute(STRING), trade_date(DATE)`（盘中触发短窗共振） |
| ETF 成分 | FMP `etf-holdings`（**仅最新、无历史** → 历史建图阶段不用，见 §4.3） | ETF → 成分 ticker + 权重 |
| 候选节点全集 | BQ `valid_ticker_v3_pure_cs` | ticker（仅 `ticker_type='CS'` 作为节点） |

### 3.1 BQ 表清单（project `mystockproject-431701`，dataset `stock_dataset`）

| 用途 | 全名 | 关键字段 |
|---|---|---|
| 日线价格 | `mystockproject-431701.stock_dataset.day_aggs_di` | `ticker(STR), open/high/low/close(FLOAT), volume/transactions(INT), trade_date(DATE)` |
| 分钟价格 | `mystockproject-431701.stock_dataset.minute_aggs_di` | 同上 + `trade_minute(STRING)` |
| 宏观结构化 | `mystockproject-431701.stock_dataset.macro_daily` | 见主方案 §5.1 A2（已 shift(+1) 对齐盘前） |
| 候选股票池 | `mystockproject-431701.stock_dataset.valid_ticker_v3_pure_cs` | `ticker, ticker_type, avg_volume` |

> 注：`trade_minute` 形如 `"2025-08-25 09:59"`（**美东时间**，需解析为 tz-aware）；`day_aggs_di` **含 ETF/指数**（SPY、QQQ 直接用作 §4.2 去市场化基准），其 `ticker` 节点全集仍以 `valid_ticker_v3_pure_cs` 的 CS 过滤；日线盘前可用至**前一交易日**收盘，`as_of` 据此截断；非复权 OHLC 的收益率按 §4.2 做拆股修正。

---

## 4. 边构建

### 4.1 Co-mention 边

滚动窗口 `W_m`（如 90 天，指数衰减）内统计共现，用 **Jaccard** 归一化消除高频偏置：

```
articles(i) = 窗口内提及 ticker i 的文章集合
co_mention(i, j) = |articles(i) ∩ articles(j)| / |articles(i) ∪ articles(j)|
```

可选改用 **PMI**（对弱关联更敏感）：

```
PMI(i, j) = log( P(i, j) / (P(i) · P(j)) )
```

要点：
- 时间衰减：第 `t` 天前的文章权重 `exp(-λ·t)`。
- ETF 展开：若文章提及某 ETF，则视为提及其 top-N 成分（按权重分摊计数），ETF 本身不入图。**注意**：FMP 成分仅有最新快照、无历史，因此**历史建图阶段禁用 ETF 展开**，仅在近实时窗口（最近数日）按当前成分展开，避免用今天的成分污染历史共现。
- 噪声过滤：剔除「一篇文章提及超过 K 个 ticker」的泛列表文（如「今日涨幅榜」），否则共现被稀释/污染。

### 4.2 Co-movement 边（去市场化）

先对每只股票日收益率做市场因子回归，取**残差收益率**，再算残差间相关：

```
r_i,t       = 个股 i 第 t 日收益率
r_mkt,t     = 市场因子收益率（首选 day_aggs_di 内的 SPY/QQQ，因表内已含 ETF/指数；备选 macro_daily.nq_futures_pct_change）
r_i,t = α_i + β_i · r_mkt,t + ε_i,t        # 对窗口 W_c 回归
co_move(i, j) = corr(ε_i, ε_j)            # 残差相关（Pearson/Spearman）
```

要点：
- 窗口 `W_c`（如 60 / 120 日），最小样本阈值（如 ≥40 个交易日）否则不建边。
- 用残差相关而非原始相关，捕捉「超越大盘的同步性」=真正的板块/主题联动。
- 只保留正相关边用于「同涨同跌」的概念聚类；负相关可单独存为「对冲/跷跷板」关系（增值，非必需）。

**非复权价格的处理（关键工程点）：**
BQ 日线为**非复权**，拆股会在收益率序列里制造 -90% 量级的伪跳，严重扭曲相关性。处理顺序：

1. **拆股修正（必须）✅ 已实现**：`dataflows.massive.fetch_splits`（`/v3/reference/splits`，分页）+ `market_tools/us/splits.py::load_splits` 拉窗口内拆股事件，`compute_returns(close, splits)` 在执行日按比例还原收益率，`build_concept_graph` 默认自动加载窗口内 splits（`splits=False` 可关闭）：
   ```
   adj_return_t = (close_t / close_{t-1}) · (split_to / split_from) − 1   # 仅拆股执行日
   # Polygon 定义：2-for-1 拆股 split_from=1, split_to=2；非复权 close 减半，
   # 乘 split_to/split_from=2 还原 → 该日收益率≈0
   ```
   > **实测**：`BN` 3:2 拆股（2025-10-10）原始日收益 −36.24% → 修正后 −4.36%；`NFLX` 10:1 拆股（2025-11-17）落入窗口时，与大科技（MSFT/AMZN/GOOGL）的去市场化相关性由被压低/转负恢复正常（GOOGL −0.048 → +0.028）。
2. **分红（可忽略）**：日分红对日收益率影响通常 <0.5%，对相关性结构无实质影响；若追求 total-return 精度，可再用 `/v3/reference/dividends` 调整。
3. **鲁棒兜底**：对极端日收益率做 winsorize（如 1%/99% 分位），或直接用 **Spearman 秩相关**，进一步抵御残留脏点。

> 备选：若不想维护拆股表，可对「近期发生过拆股的少数票」直接用 Massive Aggregates API 的 `adjusted=true` 视图重取这几只票的序列。`splits`/`dividends` 端点已标记 Deprecated，长期以 Aggregates 复权视图为准更稳妥。

### 4.3 ETF 共同归属先验边（P2 / 暂缓）

同属一个 ETF 的成分股天然有结构相关性：

```
etf_comember(i, j) = Σ_etf  min(w_etf,i, w_etf,j)      # 按持仓权重的共同归属强度
```

**降级原因**：FMP `etf-holdings` 仅提供**最新成分快照、无历史**。用今天的成分去构建历史图会引入前视偏差（fail-fast 原则下宁可不用）。因此：
- 历史建图（M1/M2）**不使用 ETF 边**，仅靠 co-mention + co-movement。
- 待后续做「当前快照视角」的近实时图时，再把 ETF 共同归属作为弱先验（`γ` 很小）叠加，且仅作用于最近窗口。
- 若将来沉淀了 ETF 成分的每日快照（自建历史），可解除该限制。

### 4.4 边融合与剪枝

```
edge_weight(i, j) = α · norm(co_mention) + β · norm(co_move) + γ · norm(etf_comember)
```

- 各分量先各自 min-max / 分位数归一化到 [0,1]。
- 默认权重建议 `α=0.4, β=0.5, γ=0.1`（可配置，后续用回测/聚类质量调参）。
- 剪枝：保留 `edge_weight ≥ θ` 或每个节点 top-K 邻居（KNN 图），控制图密度。

---

## 5. 社区发现与概念集命名

目标：**细分主题粒度** + **单 ticker 可多归属** + **簇数量受控**。标准 Louvain/Leiden 是硬划分（每节点一个簇），不满足，故采用「层次硬划分骨架 + 受控多重归属」两步法。

### 5.1 层次骨架（Hierarchical Leiden）

- 算法：**Leiden**（比 Louvain 收敛性更好），多分辨率层次聚类，产出两层：
  - **L_sector（粗）**：大板块（如 "Semiconductor"），用于稳定的高层归类。
  - **L_theme（细）**：细分主题（如 "HBM / AI 推理芯片"），是主要消费层。
- 每层各自是硬划分，但通过层级表达「板块 → 主题」的从属。

### 5.2 防膨胀四重机制（回应 G-Q4 担忧①）

1. **min_cluster_size**：细主题簇成员数 < 阈值（如 4）时，并入最近邻簇或归入 `Unclustered`，不单独成簇。
2. **簇数量软上限**：对 `resolution` 做二分搜索，使 L_theme 簇数落入目标区间（如 40–80），避免无限细化。
3. **层次截断**：对外只暴露到 L_theme，不暴露更细的叶子社区。
4. **稳定性过滤**：连续 N 日成员重叠率（Jaccard）低于阈值的簇判为噪声，丢弃或回退到其 L_sector 父簇。

### 5.3 受控多重归属（回应 G-Q4 疑问②：是，可多归属）

在硬划分骨架之上叠加带权次归属：

```
primary(i)   = Leiden 给 i 的 L_theme 簇           # is_primary=True, weight=1.0
对每个其它簇 c：
  affinity(i, c) = Σ_{j∈c} edge_weight(i, j) / Σ_j edge_weight(i, j)   # i 指向簇 c 的边权占比
  若 affinity(i, c) ≥ τ（如 0.15）→ 追加 secondary(i, c, weight=affinity)
每个 ticker 最多保留 K 个归属（如 K≤3，按 weight 取 top-K）防稀释
```

产出：`ticker → [Membership(cluster_id, weight, is_primary)]`。

### 5.4 簇命名

- 对每个 L_theme 簇取代表性成员（按加权度中心性 top-N），调用 Gemini 轻量 prompt 产出标签，结构化输出 `{cluster_id, label, theme, parent_sector}`（如成员 `MU/WDC/STX → label="存储/HBM", parent_sector="Semiconductor"`）。
- **跨日稳定性**：相邻日期的簇按成员重叠率匹配并继承旧 `cluster_id`，避免标签每日跳变；新生/消亡簇才分配/回收 id。

---

## 6. 增值能力：催化剂图传导

图不止用于聚类，还能让 L1 的个股催化剂**沿边扩散**：

```
conf_propagated(j) = conf(j) + η · Σ_i  edge_weight(i, j) · catalyst(i)
```

- 一只股票出现强催化剂（财报超预期、评级上调），按边权重提升其强相关邻居的 `catalyst_confidence`。
- 用 **一跳传导**（保守，避免过度扩散）或带阻尼的 label propagation。
- 这是静态板块表完全做不到的能力，直接增强主方案 L1/L2 的质量。

---

## 7. 存储与更新

| 对象 | 存储 | 更新频率 |
|---|---|---|
| 边表 `edges(src, dst, weight, comention_w, comovement_w, etf_w, as_of)` | BQ 表 或 本地 parquet | 每日重算 |
| 归属表 `memberships(ticker, cluster_id, weight, is_primary, as_of)` | BQ 表 或 本地 parquet | 每日/每周（多对多，支持多归属） |
| 簇标签 `clusters(cluster_id, label, theme, parent_sector, members, as_of)` | 本地 JSON / BQ | 跟随社区检测 |

- 计算引擎：**networkx**（内存图）+ pandas/numpy，规模（数千节点）完全够用，无需 Neo4j。
- 若未来节点上万或需在线查询，再评估图数据库。

---

## 8. 与主方案的接口

```python
# tradingagents/concept_graph/service.py
from pydantic import BaseModel

class Membership(BaseModel):
    cluster_id: str
    weight: float        # 隶属度，primary=1.0，secondary 为边权占比
    is_primary: bool

def get_cluster_map(as_of_date: str) -> dict[str, list[Membership]]:
    """ticker -> 多个归属（主 + 带权次），供主方案 L2 概念聚合消费。"""

def get_cluster_label(cluster_id: str) -> dict:
    """{cluster_id, label, theme, parent_sector, members}。"""

def get_neighbors(ticker: str, top_k: int = 10) -> list[tuple[str, float]]:
    """返回强相关邻居及边权重，供 L1 催化剂图传导。"""
```

主方案 L2 由「读静态映射」改为「调 `get_cluster_map()`」——注意返回是**多归属列表**，聚合时一个 ticker 可按 `weight` 同时贡献给多个 `ConceptSignal`；L1 在产出 `StockSignal` 后用 `get_neighbors()` 做一跳催化剂传导。

---

## 9. 模块与文件规划

```
tradingagents/concept_graph/
├── sources/
│   ├── comention.py      # 从新闻 tickers 数组构建共现矩阵（含 ETF 展开、衰减、泛列表过滤）
│   ├── comovement.py     # 残差收益率相关（去市场化）
│   └── etf_holdings.py    # ETF → 成分展开（FMP / BQ）
├── build_graph.py        # 边融合 + 剪枝 + networkx 图构建
├── community.py          # Louvain/Leiden 社区检测 + 跨日对齐
├── naming.py             # Gemini 簇命名（结构化输出）
├── propagate.py          # 催化剂图传导
├── store.py              # 边表/节点表/簇标签持久化（BQ/parquet/JSON）
└── service.py            # 对主方案暴露的查询接口
scripts/
└── rebuild_concept_graph.py   # 离线重算入口（Cloud Run Job / cron）
```

---

## 10. 技术选型与依赖

新增依赖：`networkx`、`python-louvain`（或 `leidenalg` + `igraph`）、`scipy`（相关/回归）。价格用 **BQ 日线表**（经 `bigquery.py`），拆股用 Massive splits；新闻复用主方案 FMP/Massive vendor；ETF 成分暂缓。

---

## 11. 开发任务拆解

| ID | 任务 | 优先级 | 预估 | 依赖 |
|----|------|--------|------|------|
| G0 | 拆股事件接入（Massive splits）+ 收益率复权修正 ✅ 已完成并实测 | P0 | 0.5d | - |
| G1 | co-movement 边（BQ 日线 → 拆股修正 → 去市场化残差相关 + winsorize/Spearman） | P0 | 1d | G0 |
| G2 | co-mention 边（Jaccard/PMI + ETF 展开 + 衰减 + 过滤） | P0 | 1d | 新闻 vendor |
| G3 | 边融合 + 剪枝 + networkx 图构建 | P1 | 0.5d | G1,G2 |
| G4 | 层次 Leiden + 防膨胀（min_size/簇上限二分/截断）+ 多重归属 ✅ 已实现（稳定性过滤/跨日对齐属 M3，依赖历史快照） | P1 | 1.5d | G3 |
| G5 | 簇命名（Gemini 结构化输出 → 填充 label/parent_sector）✅ 已实现（`naming.py`，默认 gemini-3.1-flash-lite） | P2 | 0.5d | G4 |
| G6 | 催化剂图传导 `propagate.py` | P2 | 0.5d | G3 |
| G7 | 持久化（本地 JSON `store.py`）+ `service.py` 接口（get_cluster_map/label/neighbors）✅ 已实现 | P1 | 0.5d | G3,G4 |
| G8 | `scripts/rebuild_concept_graph.py` 离线重算（CLI + 日志抑制 + 命名 + `--as-of latest` + `--gcs-bucket` 上传）✅ 已实现；`scripts/run_concept_graph_daily.sh` 每日 wrapper；调度器配置待用户确认 | P2 | 0.5d | G7 |

> 备注（G-Q3）：G2 开发新闻 vendor 时，**增加历史回溯深度测试**——验证 Massive/FMP 新闻接口能否拉到 ≥2024 的历史（co-mention 90 天窗口需足够历史冷启动）；若某接口历史不足，则该信号的建图起点相应顺延，或先用较短窗口启动。

**里程碑**：
- M1（图能建起来）：G0+G1+G2+G3
- M2（概念集可用）：G4+G7 → 主方案 L2 可消费 `get_cluster_map()`
- M3（增值）：G5+G6+G8

---

## 12. 待探查 / 待确认

| ID | 事项 | 状态 |
|----|------|------|
| G-Q1 | ETF 成分来源 | ✅ **已定**：FMP `etf-holdings`（仅最新），历史建图暂不用，整体降级 P2 |
| G-Q2 | 价格数据源 | ✅ **已定**：BQ 日线表（非复权）+ Massive splits 修正；分钟表用于盘中 |
| G-Q3 | 历史覆盖深度 | ✅ **已定**：日线 ≥2024 起（足够 co-movement 窗口与滚动）；新闻历史未测 → 在新闻 vendor 探查阶段**增加历史回溯深度测试**（见 §11 任务备注） |
| G-Q4 | 簇粒度 + 多归属 | ✅ **已定**：细分主题（L_theme 层）+ 防膨胀四重机制 + 单 ticker 多归属（K≤3），见 §5 |
| G-Q5 | BQ 价格表名 + project/dataset | ✅ **已定**：日线 `day_aggs_di`、分钟 `minute_aggs_di`（完整全名与字段见 §3.1） |

---

## 13. 风险与注意

- **非复权拆股伪跳**：必须用 splits 事件修正收益率，否则拆股日的 -90% 伪收益会摧毁该股相关性——与去市场化并列为两大必修工程点。
- **伪相关**：不做去市场化会得到「万物相关」的退化图——这是本方案最关键的工程点。
- **泛列表新闻污染**：涨跌幅榜类文章会制造大量虚假共现，必须按「单篇 ticker 数」过滤。
- **簇漂移**：社区 id 每日不稳定，需跨日成员对齐，否则下游标签抖动。
- **冷启动**：新股/小票样本不足，按阈值留空而非强行建边（fail-fast）。
- **前视偏差**：建图所用新闻与价格须严格 `as_of_date` 截断，盘前任务只能用昨日及更早数据。
- **成本**：簇命名调用 LLM 的次数 = 簇数量级（小），可忽略；图计算为 CPU 密集但单机可承载。
