# 方向许可层（LLM 基本面与宏观风控层）方案文档

> 版本：v0.2（草案）
> 定位：在现有 TradingAgents 多 Agent 框架之上，新增一个**三层（个股 → 概念集 → 大盘）的宏观/基本面闸门层**，作为系统的「战略指挥官」与「物理断路器」。
> v0.2 变更：吸收用户对 D1–D6 的决策；架构由「大盘 regime + 白名单」升级为**自底向上的三层聚合**；凭证统一走 **GCP Secret Manager**；数据源分工落定（FMP=宏观/大盘，Massive=个股/实时）。

---

## 1. 背景与目标

### 1.1 定位

- **战略指挥官**：从宏观与基本面维度判断当前处于「顺风局（Bullish）/ 震荡（Range）/ 逆风局（Bearish）」，剥离微观技术噪音。
- **物理断路器**：过滤系统性暴跌风险与个股重大利空，生成允许量化模型开仓的「许可集」。**双向**——可做多 / 可做空 / 禁止交易。

### 1.2 输入

| 类别 | 内容 | 来源（已落定） |
|---|---|---|
| 宏观日历预期 | 非农、CPI、美联储决议 | **FMP** `/stable/economic-calendar` |
| 大盘 / 宏观新闻 | 货币政策、地缘、市场情绪 | **FMP** `/stable/news/general-latest` |
| 个股非结构化文本 | 财报、突发催化剂、投行评级/目标价 | **Massive(Polygon)** `/v2/reference/news` + **Benzinga** `/benzinga/v2/news` |
| 宏观结构化数据 | 利率 / VIX 趋势 | **BigQuery** `macro_daily` 表 |
| 候选股票池 | 待判定的个股清单 | **BigQuery** `valid_ticker_v3_pure_cs` 表 |
| 概念集 / 板块映射 | ticker → 芯片/存储等概念 | 待定（见探查清单 P3） |

### 1.3 输出（三层结构）

一个结构化、可解析的 JSON（先落地为**本地 JSON 文件**）：

- **L3 大盘层**：大盘短期状态（Bullish / Range / Bearish）+ 宏观综述。
- **L2 概念集层**：各概念/板块（如芯片、存储）的强弱判断 + 依据。
- **L1 个股层**：候选池中每只个股的方向许可（Long / Short / Block）+ 催化剂置信度（0–1）+ 依据。

> 大盘 regime 作为断路器，对概念集与个股许可进行**自上而下的约束/否决**；个股与概念信号则**自下而上**为大盘判断提供佐证。

---

## 2. 现有框架分析（可复用资产）

TradingAgents 是基于 **LangGraph** 的多 Agent 框架，核心工作流为**单票、单日**决策：

```
分析师团队(基本面/情绪/新闻/技术) → 多空研究员辩论 → Trader → 风控三辩 → Portfolio Manager → Buy/Sell/Hold
```

入口（`tradingagents/graph/trading_graph.py`）：

```python
ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
```

本方案将**复用**以下基础设施：

| 复用资产 | 文件 | 用途 |
|---|---|---|
| LLM 工厂（含 Gemini） | `tradingagents/llm_clients/factory.py` / `google_client.py` | 直接用 Gemini 3.1 Pro |
| 数据 vendor 路由 | `tradingagents/dataflows/interface.py` | 插入 FMP / Massive 新闻源 |
| 新闻工具封装 | `tradingagents/agents/utils/news_data_tools.py` | 复用 `get_news` / `get_global_news` |
| 结构化输出机制 | `tradingagents/agents/schemas.py` + `agents/utils/structured.py` | 稳定可解析 JSON（Gemini 走 response_schema） |
| 配置与环境变量覆盖 | `tradingagents/default_config.py` | `TRADINGAGENTS_*` / `data_vendors` |
| API key 映射 | `tradingagents/llm_clients/api_key_env.py` | 注册新数据源 key |

> 关于「更充分利用现有框架」（用户要求 2）：L1 个股层的判断逻辑与现有「分析师 + 工具」模式高度同构，可直接复用 analyst 节点风格（甚至对高优先级个股调用简化版逐票分析）。L2/L3 是在其上新增的聚合层。

---

## 3. 架构落差与核心设计决策

### 3.1 落差分析

| 维度 | 现有框架 | 方向许可层 |
|---|---|---|
| 分析对象 | 单只股票 | 候选池个股 + 概念集 + 大盘（三层） |
| 输出 | 单票 Buy/Hold/Sell | 三层结构化 JSON |
| 触发方式 | 手动逐票 `propagate` | 盘前批处理 + 盘中特殊事件触发 |
| 数据类别 | 股价/技术/基本面/新闻 4 类 | 额外需要宏观结构化时序(BQ) + 宏观日历 + 概念映射 |
| 定位 | 微观交易决策 | 宏观闸门 / 断路器 |

### 3.2 核心决策

**决策 1：新建独立编排流，三层自底向上聚合。**
方向许可层是现有框架的**上游层**。三层关系：

```
L1 个股信号(候选池逐票)  ──聚合──▶  L2 概念集强弱(芯片/存储/…)  ──聚合──▶  L3 大盘 regime
        ▲                                                                      │
        └──────────────── 大盘 regime 断路器约束(自上而下否决) ◀───────────────┘
```

**决策 2：复用「水管」，新建「业务层」。**
LLM 工厂、vendor 路由、结构化输出、新闻工具全部复用；新增数据源适配器、三层 Schema、三层编排 Agent、调度脚本。

**决策 3：凭证统一走 GCP Secret Manager（用户要求 1）。**
所有 API key（Gemini、FMP、Massive）与 BQ 凭证均从 Secret Manager 读取，不落明文 `.env`。新增一个轻量 secret 读取模块，在进程启动时把 secret 注入为环境变量，从而沿用框架现有的「env var → key」约定，改动面最小。

**决策 4：数据源分工。**
- 宏观/大盘（日历 + general news）→ **FMP**
- 个股催化剂/财报/评级 + 盘中实时 → **Massive(Polygon/Benzinga)**

**决策 5：输出先本地 JSON。** 整套系统可部署后再切换/追加 BQ 结果表。

---

## 4. 目标架构

```mermaid
flowchart TD
    subgraph Secret[GCP Secret Manager]
        SM[(secrets: gemini / fmp / massive / bq)]
    end

    subgraph Sources[数据源]
        FMP[FMP\n经济日历 + general news]
        MAS[Massive/Polygon\n个股 + Benzinga 实时]
        BQ[(BigQuery\nmacro_daily + 候选池表)]
    end

    subgraph Gate[方向许可层 - 新建]
        VBQ[bigquery 数据模块]
        VN[fmp / massive vendor 适配器]
        T1[get_macro_daily]
        T2[get_economic_calendar]
        T3[get_global_news / get_news]
        L1[L1 个股信号节点]
        L2[L2 概念集聚合节点]
        L3[L3 大盘 regime 节点 + 断路器]
        SCHEMA[[RegimeReport JSON Schema]]
    end

    subgraph Out[输出]
        RES[本地 JSON 文件]
        TA[现有 TradingAgents 逐票分析(可选)]
    end

    SM -.注入 env.-> Gate
    BQ --> VBQ --> T1 --> L3
    BQ --> L1
    FMP --> VN --> T2 --> L3
    FMP --> T3 --> L3
    MAS --> VN
    VN --> L1
    L1 --> L2 --> L3 --> SCHEMA --> RES
    RES -->|许可集| TA

    SCHED[Cloud Scheduler / cron\n盘前 + 盘中特殊触发] -.-> Gate
```

---

## 5. 模块详细设计

### 5.1 模块 A：数据接入层

> **市场化 tools 分层（已落地）**：数据接入已从 `concept_graph` 解耦，独立为可切换的市场工具层 `tradingagents/market_tools/`。
> - `market_tools/__init__.py`：定义市场无关契约 `MarketDataTools`（Protocol）+ `get_market_tools(market="US")` 解析器。「换市场 = 换工具 = 换数据源」。
> - `market_tools/us/`：美股实现。`news.py`（委托 `dataflows.massive`/`fmp`，复用已注册的 vendor 适配器）、`prices.py`（`day_aggs_di`/`minute_aggs_di`）、`macro.py`（`macro_daily` + LLM 摘要）、`universe.py`（候选池）、`_bigquery.py`（BQ 公共 helper，ADC 鉴权）。
> - `concept_graph/sources/` 仅保留纯算法建边器（co-mention / co-movement），消费 DataFrame，不再持有取数代码。
> - 其他市场只需新增 `market_tools/<mkt>/` 实现同一契约即可复用整条 regime/concept 流水线。

#### A1. 新闻 vendor 适配器（FMP + Massive）
- 新建 `tradingagents/dataflows/fmp.py`：
  - `get_global_news(curr_date, look_back_days, limit)` → `GET /stable/news/general-latest`
  - `get_economic_calendar(from_date, to_date, country="US")` → `GET /stable/economic-calendar`（≤90 天）
  - 可选 `get_news(ticker, start_date, end_date)` → `/stable/news/stock`
- 新建 `tradingagents/dataflows/massive.py`：
  - `get_news(ticker, start_date, end_date)` → `GET /v2/reference/news`（带 sentiment）
  - 可选 `get_benzinga_news(ticker, since)` → `GET /benzinga/v2/news`（评级/目标价/财报，盘中实时）
- 在 `dataflows/interface.py` 注册 `VENDOR_LIST` 与 `VENDOR_METHODS`（`get_news` → massive；`get_global_news` → fmp）。
- 启用：`config["data_vendors"]["news_data"]` 按层指定，或用 `tool_vendors` 做工具级路由。
- **实现前需 fetch 字段级文档**（见探查清单 P1/P2）。

#### A2. BigQuery 数据模块 ✅ 已实现（market_tools/us）
- 落点：`market_tools/us/{prices,macro,universe}.py` + 公共 `market_tools/us/_bigquery.py`（依赖 `google-cloud-bigquery`）。
- 鉴权：ADC（与 Secret Manager 一致）。
- **已实测**（py10，project `mystockproject-431701`）：候选池 610 只、`macro_daily` 取数+摘要、`day_aggs_di` 日线、`minute_aggs_di` 分钟全部跑通。
- **`macro_daily` 表字段（已提供）**：

  | 字段 | 类型 | 含义 |
  |---|---|---|
  | `trade_date` | DATETIME | 交易日（主键，**已 shift(+1) 对齐到盘前可用**） |
  | `us10y_yield` | FLOAT | 美 10 年期国债收益率 |
  | `us_yield_curve_spread` | FLOAT | 10Y-2Y 利差（衰退指标） |
  | `vix_close` | FLOAT | VIX 收盘 |
  | `nq_futures_close` | FLOAT | 纳指期货收盘 |
  | `dxy_close` | FLOAT | 美元指数 |
  | `vix_pct_change` | FLOAT | VIX 日变化率 |
  | `nq_futures_pct_change` | FLOAT | 纳指期货日变化率 |

  > 数据由 FRED 经 `docs/macro_daily_deploy.py`（Cloud Functions）落库；管道内 `shift(1, freq='D')` 已把每行特征对齐为「该交易日盘前可见的上一日收盘值」，因此读当日行**无未来函数风险**。`nq_futures_pct_change` 亦可作为概念图谱 co-movement 的去市场化基准。

- 函数：
  - `get_macro_daily(curr_date, look_back_days) -> str`：查询 `macro_daily`，返回利率/VIX/利差/DXY 近 N 日趋势摘要（格式化为便于 LLM 消费的文本/小表）。
  - `get_candidate_universe(filters) -> list[str]`：查询候选池（默认 + 可放宽口径）：
    ```sql
    select ticker
    from `mystockproject-431701.stock_dataset.valid_ticker_v3_pure_cs`
    where ticker_type = 'CS'
      and avg_volume >= 3000000   -- 默认 5M，可放宽至 3M（参数化）
      and avg_price  >= 5         -- 排除低价小额股
    ```
    `avg_volume` 阈值与是否限制 `ticker_type` 做成可配置参数。
  - `get_daily_prices(tickers, start_date, end_date) -> DataFrame`：查询 BQ 日线价格表（`ticker/open/high/low/close/volume/transactions/trade_date`，非复权），供概念图谱 co-movement 使用（拆股修正见 `concept-graph-design.md` §4.2）。分钟表 `get_minute_prices(...)` 供盘中触发。
- 新工具 `get_macro_daily`（仿 `news_data_tools.py` 的 `@tool` 写法）绑定到 L3 节点。

#### A3. Secret Manager 接入（用户要求 1）✅ 已实现
- `tradingagents/dataflows/secrets.py`：
  - 依赖 `google-cloud-secret-manager`，鉴权走 ADC（本地 `gcloud auth application-default login`，线上用 Cloud Run 服务账号）。
  - `get_secret(secret_id, project_id, version)`：读取单个 secret（自动 strip 尾部换行）。
  - `load_secrets_to_env(mapping, override=False)`：把 secret 注入 `os.environ`，默认不覆盖已存在的 env（便于测试），复用框架现有 env-var 约定。
  - 默认映射 `DEFAULT_SECRET_ENV_MAP`：`massive_key→MASSIVE_API_KEY`、`massive_key_md5→MASSIVE_KEY_MD5`、`fmp_api_key→FMP_API_KEY`、`fred_api_key→FRED_API_KEY`、`gemini_api_key→GOOGLE_API_KEY`。
  - 本地零存储：vendor 模块仍只读 env，真实 key 仅在内存中存在。
  - **已实测**：`load_secrets_to_env` + Massive/FMP 实时接口在 py10 环境跑通（行情新闻、个股新闻、全球新闻、经济日历）。

### 5.2 模块 B：LLM 输出契约（三层 Pydantic Schema）

新建 `tradingagents/regime/schemas.py`：

```python
from enum import Enum
from pydantic import BaseModel, Field

class MarketRegime(str, Enum):
    BULLISH = "Bullish"
    RANGE   = "Range"
    BEARISH = "Bearish"

class Direction(str, Enum):
    LONG  = "Long"
    SHORT = "Short"
    BLOCK = "Block"

class Strength(str, Enum):
    STRONG  = "Strong"
    NEUTRAL = "Neutral"
    WEAK    = "Weak"

# L1 个股层
class StockSignal(BaseModel):
    ticker: str
    direction: Direction
    catalyst_confidence: float = Field(description="0-1，催化剂置信度")
    reason: str

# L2 概念集层
class ConceptSignal(BaseModel):
    concept: str = Field(description="概念/板块名，如 Semiconductor、Memory")
    strength: Strength
    member_tickers: list[str]
    rationale: str

# L3 大盘层 + 顶层聚合
class RegimeReport(BaseModel):
    as_of_date: str
    market_state: MarketRegime
    macro_summary: str = Field(description="宏观与基本面综述，剥离微观噪音")
    concept_signals: list[ConceptSignal]
    stock_signals: list[StockSignal]
```

> 复用 `agents/utils/structured.py` 的 `bind_structured()` + `invoke_structured_or_freetext()`，弱模型/异常优雅回退；Gemini 原生 `response_schema` 保证 JSON 稳定。

### 5.3 模块 C：三层编排 Agent

新建 `tradingagents/regime/`：
- `l1_stock.py`：对候选池逐票（或分批）调用，绑定个股新闻工具（Massive/Benzinga）+ 基本面，产出 `StockSignal`。复用 analyst 节点风格。
- `l2_concept.py`：**调用概念图谱子系统** `concept_graph.service.get_cluster_map()` 取动态概念集（替代静态映射），聚合 L1 信号产出 `ConceptSignal`；并可用 `get_neighbors()` 对 L1 做催化剂图传导。详见 `concept-graph-design.md`。
- `l3_regime.py`：绑定 `get_macro_daily` / `get_economic_calendar` / `get_global_news`，结合 L2 信号产出 `RegimeReport`，并执行**断路器**（regime=Bearish/Range 时收紧或否决 long）。
- `commander.py`：LangGraph 编排上述三层节点。

**Prompt 设计要点**：角色=战略指挥官+断路器；先宏观后微观；双向；置信度 0–1 给明确锚点；大盘 regime 对个股/概念有最终否决权。

### 5.4 模块 D：批处理调度

- 新建 `scripts/run_regime_gate.py`：
  1. Secret Manager 注入凭证。
  2. 取候选池（BQ）+ 宏观结构化（BQ）+ 日历/新闻（FMP/Massive）。
  3. 运行三层 commander → `RegimeReport`。
  4. 写**本地 JSON**（如 `~/.tradingagents/regime/<date>.json`）。
- 调度：
  - **盘前**：Cloud Scheduler / cron 每日定时。
  - **盘中特殊触发**：财报、美联储讲话等事件触发（可由 Benzinga 实时流或事件回调驱动单票/单概念的增量重算）。

### 5.5 模块 E（可选）：与下游打通

- 把 `stock_signals` 中 `direction != Block` 的 ticker 传入现有 `TradingAgentsGraph.propagate()` 做逐票深度分析。
- regime=Range 或个股 Block 时跳过下游（断路器生效）。

---

## 6. 开发任务拆解

| ID | 任务 | 模块 | 优先级 | 预估 | 依赖 |
|----|------|------|--------|------|------|
| T0 | Secret Manager 接入模块 ✅ | A3 | P0 | 0.5d | 已完成并实测 |
| T1 | 切换 Gemini 3.1 Pro（配置/env） | LLM | P0 | 0.5h | T0 |
| T2 | `market_tools/us`：macro_daily + 候选池 + 日/分钟价格 ✅ | A2 | P0 | 1d | 已完成并实测 |
| T3 | `regime/schemas.py`（三层 Schema） | B | P0 | 0.5d | - |
| T4 | FMP vendor（日历 + general news） | A1 | P1 | 0.5d | P1 字段文档 |
| T5 | Massive vendor（个股 + Benzinga） | A1 | P1 | 0.5d | P2 字段文档 |
| T6 | 概念集接入（消费知识图谱子系统） | A2/A1 | P1 | 0.5d | 概念图谱子方案 M2 |
| T7 | L1/L2/L3 + commander 编排 | C | P1 | 2d | T2–T6 |
| T8 | `run_regime_gate.py` + 本地 JSON 落盘 | D | P1 | 0.5d | T7 |
| T9 | 盘前调度 + 盘中事件触发 | D | P2 | 1d | T8 |
| T10 | 与现有逐票 graph 打通 | E | P2 | 0.5d | T8 |

---

## 7. 决策点（已落定）

| ID | 决策点 | 结论 |
|----|--------|------|
| D1 | 宏观日历数据源 | **FMP** `/stable/economic-calendar`（已探查） |
| D2 | 个股 + 大盘新闻源 | 个股=**Massive/Benzinga**；大盘=**FMP general news**（已探查，缺口已补） |
| D3 | 候选股票池来源 | **BQ** `valid_ticker_v3_pure_cs`，过滤条件可放宽并参数化 |
| D4 | 结果输出落点 | **先本地 JSON**，可部署后再加 BQ 结果表 |
| D5 | Gemini 型号 | **gemini-3.1-pro**（deep）+ flash（quick，可选） |
| D6 | 触发时点 | **盘前为主** + 盘中特殊事件（财报/美联储讲话）触发 |

---

## 8. 探查 / 待用户提供清单

| ID | 事项 | 谁来做 | 状态 |
|----|------|--------|------|
| P1 | FMP 经济日历 + general news 字段级文档 | 我（WebFetch 公开文档） | 已基本确认，实现时补字段 |
| P2 | Massive `/v2/reference/news` + Benzinga 字段 + 你的订阅档位 + **历史回溯深度测试（能否拉到 ≥2024）** | 我探查 + **你确认档位/权限** | 待确认订阅是否含 Benzinga/WebSocket，以及历史窗口 |
| P3 | 概念集/板块映射来源（ticker→芯片/存储…） | ✅ **方案已定**：知识图谱动态涌现，见 `concept-graph-design.md` | 升级为子系统 |
| P4 | BQ `macro_daily` 表 schema | ✅ **已提供**（见 §5.1 A2 字段表） | 已完成 |
| P5 | 候选池过滤口径 | ✅ **已定**：默认 `ticker_type='CS'`；可选放宽 `avg_volume≥3,000,000` 且 `avg_price≥5`（排除低价小额股） |
| P6 | Secret Manager secret id | ✅ **已定**：`gemini_api_key/fmp_api_key/massive_key/massive_key_md5/fred_api_key`（`massive_key_md5` 用途待确认） |
| P7 | FMP / Massive 订阅档位 | ✅ **已定**：Massive option+stock 均最高档（含 Benzinga/WebSocket）；FMP **starter**（可升级）→ 需核实 economic-calendar 历史 / general news 是否在档 |
| P3' | 概念集映射 → 升级为知识图谱子系统 | ✅ 方案已出：见 `concept-graph-design.md` | 见子方案 |

---

## 9. 配置与凭证（统一走 Secret Manager）

```text
GCP Secret Manager（project: mystockproject-431701）— secret id 已确认
├── gemini_api_key    → 注入 GOOGLE_API_KEY
├── fmp_api_key       → 注入 FMP_API_KEY
├── massive_key       → 注入 MASSIVE_API_KEY
├── massive_key_md5   → 注入 MASSIVE_KEY_MD5（用途待确认：签名/缓存键？见下方备注）
├── fred_api_key      → 注入 FRED_API_KEY（供 macro_daily 落库管道 macro_daily_deploy.py）
└── （BQ 用 ADC / 服务账号）→ GOOGLE_APPLICATION_CREDENTIALS

运行期配置（非密钥）：
TRADINGAGENTS_LLM_PROVIDER=google
TRADINGAGENTS_DEEP_THINK_LLM=gemini-3.1-pro
BQ_PROJECT=mystockproject-431701   BQ_DATASET=stock_dataset
BQ_MACRO_TABLE=macro_daily         BQ_UNIVERSE_TABLE=valid_ticker_v3_pure_cs
BQ_DAY_TABLE=day_aggs_di           BQ_MINUTE_TABLE=minute_aggs_di
```

> 备注 `massive_key_md5`：Polygon/Massive 标准鉴权只需单个 key（Bearer 或 `apiKey`）。该 md5 可能是 key 的指纹/缓存键或某接口的签名，**接入时确认其用途**，默认仅注入 `massive_key`。

新增依赖（`pyproject.toml`）：`google-cloud-bigquery`、`google-cloud-secret-manager`。

---

## 10. 落地路线图

- **里程碑 1（地基）**：T0 + T1 + T2 + T3 —— Secret Manager、Gemini、BQ 取数、三层 Schema。
- **里程碑 2（数据源）**：T4 + T5 + T6 —— FMP / Massive vendor + 概念映射。
- **里程碑 3（最小闭环）**：T7 + T8 —— 三层编排产出 `RegimeReport` 并落本地 JSON。
- **里程碑 4（增强）**：T9 + T10 —— 盘前调度 + 盘中触发 + 打通下游逐票分析。

> 并行子线：概念图谱子系统（`concept-graph-design.md`）可独立推进，其 M2 产出 `get_cluster_map()` 是主方案 T6 的前置依赖。

---

## 11. 风险与注意事项

- **凭证管理**：全部走 Secret Manager；Gemini key ≠ BQ 服务账号，分开存储。
- **LLM 输出漂移**：强制 `response_schema` + 回退；置信度需明确打分锚点。
- **成本与速率**：候选池可能很大；区分 deep/quick 模型，按概念分批，叠加缓存以适配 FMP/Massive 速率限制。
- **数据时效 / 未来函数**：`macro_daily` 已在落库管道内 shift(+1) 对齐盘前（无未来函数）；但**日历、新闻**仍须按 `curr_date` 严格截断，盘前任务避免 look-ahead bias。
- **断路器优先级**：大盘 regime 优先于个股催化剂；系统性风险触发时直接清空/否决 long 许可。
- **盘中触发幂等**：特殊事件触发的增量重算需可幂等覆盖当日 JSON，避免重复/冲突。
- **价格表口径**：`day_aggs_di` 含 ETF/指数（SPY、QQQ 可直接作 co-movement 去市场化基准）；盘前可用至**前一交易日**收盘，`as_of` 据此截断；`minute_aggs_di.trade_minute` 形如 `"2025-08-25 09:59"`（美东时间）。
