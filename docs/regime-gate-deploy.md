# Concept Graph / Regime Gate — Cloud Run 部署

每日盘前批处理拆成**两个独立的 Cloud Run Job**,各由 **Cloud Scheduler** 在美东工作日定时触发,结果写入 GCS:

1. **`concept-graph-daily`** — 重建当天会话的概念图,上传 `gs://trading_agent/concept_graph/<session>/`。
2. **`regime-daily`** — 从 GCS 拉回同一会话的概念图快照,跑 regime gate 级联,上传 `gs://trading_agent/regime_gate/<session>/`。

> 拆分动机:概念图与 regime gate 的算力/失败域/调度节奏不同,合在一个 Job 里耦合过紧(概念图改动要连带重跑 gate)。拆开后概念图可独立调度、独立排错、独立回填(`scripts/rollback_concept_graph.py`),gate 只消费 GCS 上已落地的快照。
>
> 时区:Scheduler 用 `America/New_York`,自动处理夏令时,无需手动调时。

## 1. 架构

```
Cloud Scheduler (09:00 ET, Mon–Fri)        Cloud Scheduler (09:10 ET, Mon–Fri)
        │  HTTP :run                                │  HTTP :run
        ▼                                           ▼
Cloud Run Job  concept-graph-daily          Cloud Run Job  regime-daily
  (entrypoint_concept_graph.sh)               (entrypoint_regime.sh)
        │  rebuild_concept_graph.py                  │  run_regime_gate.py --cg-from-gcs
        │    --as-of latest --all --name             │    --as-of latest
        ▼                                           ▼
gs://trading_agent/concept_graph/<session>/ ──读──▶ gs://trading_agent/regime_gate/<session>/
        ▲                                           ▲
        └──────── BigQuery (价格/macro_daily) · Secret Manager (keys) · GCS ────────┘
```

- **依赖与时序**:`regime-daily` 用 `--cg-from-gcs` 下载**同一 session** 的概念图快照,因此必须排在 `concept-graph-daily` **之后**(默认 09:00 → 09:10,留 10 分钟余量)。若当天快照尚未生成,gate 会 `FileNotFoundError` 直接失败(fail-fast,不会用旧图)。
- `--as-of latest` = 今天 ET(交易会话日),数据仍用上一交易日收盘 → 无前视泄露。
- 两个 Job 共用**同一镜像**,仅靠 `--args` 选择入口脚本(见下)。
- 本地产物写 `/tmp`(Cloud Run 只读文件系统,仅 `/tmp` 可写);durable 副本是 GCS 上传。日志走 Cloud Logging。

## 2. 部署件

| 文件 | 作用 |
|---|---|
| `deploy/Dockerfile` | 批处理镜像:`pip install .[regime]`;`ENTRYPOINT ["bash"]`,默认 `CMD` 指向 regime 入口,概念图 Job 用 `--args` 覆盖 |
| `deploy/entrypoint_concept_graph.sh` | **概念图 Job 入口**:只跑 `rebuild_concept_graph.py`,产物 `/tmp`,上传 GCS |
| `deploy/entrypoint_regime.sh` | **regime Job 入口**:`run_regime_gate.py --cg-from-gcs`(先从 GCS 拉概念图),上传 GCS |
| `deploy/cloudbuild.yaml` | 用 `deploy/Dockerfile` 构建并推 Artifact Registry |
| `.dockerignore` | 把 `.git`/输出目录/日志排除出构建上下文 |

依赖(`pyproject.toml` 的 `[regime]` extra)关键项:`google-cloud-bigquery` + **`google-cloud-bigquery-storage`**(快速拉取)+ **`db-dtypes`** + **`pyarrow`**(`to_dataframe` 必需)+ `google-cloud-secret-manager` + `google-cloud-storage` + **`google-genai`**(`google_client` 直接 import)+ `networkx`/`leidenalg`/`python-igraph`(图计算)。

## 3. 前置条件

- **FMP 套餐:Premium 及以上**。point-in-time 基本面用**季度**报表(`income/balance/cash-flow-statement?period=quarter`)+ `acceptedDate`,Starter 只有年度,不够。
- **API keys** 全在 Secret Manager(`secrets.py` 的 `DEFAULT_SECRET_ENV_MAP`:`fmp_api_key`/`massive_key`/`gemini_api_key`/...),代码运行时注入环境变量,不落地。
- **BigQuery** 价格表 + `macro_daily`(由数据管道维护)在触发时应已就绪当天可见行。

## 4. 首次部署(一次性)

```bash
export PROJECT=mystockproject-431701
export REGION=us-central1
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/regime/regime-job:latest
export SA=regime-job@$PROJECT.iam.gserviceaccount.com
gcloud config set project $PROJECT

# 4.1 开 API
gcloud services enable run.googleapis.com cloudscheduler.googleapis.com \
  cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

# 4.2 Artifact Registry 仓库
gcloud artifacts repositories create regime --repository-format=docker --location=$REGION

# 4.3 构建并推镜像
gcloud builds submit --config deploy/cloudbuild.yaml --substitutions _IMAGE=$IMAGE

# 4.4 运行时服务账号 + 角色
gcloud iam service-accounts create regime-job --display-name "Regime gate daily job"
for ROLE in roles/bigquery.dataViewer roles/bigquery.jobUser roles/bigquery.user \
            roles/secretmanager.secretAccessor roles/storage.objectAdmin \
            roles/run.developer; do
  gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:$SA" --role="$ROLE"
done

# 4.5 创建两个 Job(共用同一镜像,靠 --args 选入口脚本)
#     概念图 Job:覆盖默认 CMD 指向概念图入口
gcloud run jobs create concept-graph-daily \
  --image $IMAGE --region $REGION --service-account $SA \
  --command bash --args /app/deploy/entrypoint_concept_graph.sh \
  --set-env-vars RG_GCS_BUCKET=trading_agent,GOOGLE_CLOUD_PROJECT=$PROJECT \
  --memory 4Gi --cpu 2 --max-retries 1 --task-timeout 7200
#     regime Job:用镜像默认入口(entrypoint_regime.sh)即可,无需 --args
gcloud run jobs create regime-daily \
  --image $IMAGE --region $REGION --service-account $SA \
  --set-env-vars RG_GCS_BUCKET=trading_agent,GOOGLE_CLOUD_PROJECT=$PROJECT \
  --memory 4Gi --cpu 2 --max-retries 1 --task-timeout 7200

# 4.6 手动验证(先概念图,再 regime;regime 依赖当天概念图快照已在 GCS)
gcloud run jobs execute concept-graph-daily --region $REGION --wait
gcloud run jobs execute regime-daily --region $REGION --wait

# 4.7 Scheduler:概念图 09:00 ET,regime 09:10 ET(开盘前、留 10 分钟给概念图先落地);时区自动管 DST
gcloud scheduler jobs create http concept-graph-daily-trigger --location $REGION \
  --schedule "0 9 * * 1-5" --time-zone "America/New_York" \
  --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/concept-graph-daily:run" \
  --http-method POST --oauth-service-account-email $SA
gcloud scheduler jobs create http regime-daily-trigger --location $REGION \
  --schedule "10 9 * * 1-5" --time-zone "America/New_York" \
  --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/regime-daily:run" \
  --http-method POST --oauth-service-account-email $SA
```

> 已有旧的合并版 `regime-daily`(概念图+gate)?改造方式:重建镜像后
> `gcloud run jobs update regime-daily --image $IMAGE --region $REGION`(默认入口已切到只跑 gate 的 `entrypoint_regime.sh`),再按 4.5 新建 `concept-graph-daily` 与其 Scheduler 即可。

所需 IAM 角色一览:

| 角色 | 为什么 |
|---|---|
| `roles/bigquery.dataViewer` + `roles/bigquery.jobUser` | 查询价格/macro 表 |
| `roles/bigquery.user` | `bigquery.readsessions.create`(Storage Read API 快速拉取);缺了会 403 |
| `roles/secretmanager.secretAccessor` | 读 API keys |
| `roles/storage.objectAdmin` | 写 GCS 产物 |
| `roles/run.developer` | Scheduler 触发 Job(`run.jobs.run`) |

## 5. 更新后重新部署(Agent 代码改动后)

代码改动后**只需重建镜像 + 让两个 Job 都指向新镜像 + 跑**,Scheduler 不用动:

```bash
export PROJECT=mystockproject-431701 REGION=us-central1
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/regime/regime-job:latest

gcloud builds submit --config deploy/cloudbuild.yaml --substitutions _IMAGE=$IMAGE && \
gcloud run jobs update concept-graph-daily --image $IMAGE --region $REGION && \
gcloud run jobs update regime-daily        --image $IMAGE --region $REGION && \
gcloud run jobs execute concept-graph-daily --region $REGION --wait && \
gcloud run jobs execute regime-daily        --region $REGION --wait   # 可选:手动验证一次
```

> 两个 Job 共用同一镜像,所以都要 `jobs update`(同 tag 也必须,否则仍指向旧镜像摘要)。
> 新增 Python 依赖 → 加进 `pyproject.toml` 的 `[regime]` extra,再重建即可。

## 6. 模型与并发调优

`run_regime_gate.py` / `commander.py` 支持**分层模型**(深度算力只留给最高杠杆的 L3):

| 参数 | 默认 | 层 |
|---|---|---|
| `--l1-model` | `gemini-3-flash-preview` | L1 逐票(高频,~数百次) |
| `--concept-model` | `gemini-3.1-pro-preview` | L2 簇/板块 |
| `--regime-model` | `gemini-3.1-pro-preview` | L3 大盘(深度思考) |
| `--model` | — | 设了则一键覆盖三层 |
| `--max-workers` | 6 | LLM 并发(受 Gemini RPM 限制) |

**实测(2026-06-11)依据**:L1/L2 全 flash vs 全 Pro,L3 大盘判断一致(都 Bearish),但 L2 板块 6/11 不同、个股方向一致率 ~83%。故默认定为 **L1=flash、L2=Pro、L3=Pro**:L1 是耗时大头(~数百次),用 flash 吃下大部分提速;L2(板块/主题,仅数十次)与 L3 保持 Pro 以保住概念层与大盘质量。若想更省可把 `--concept-model` 也设为 flash。

`entrypoint_regime.sh` 不传这些参数 → 用默认值;要改部署口径,在 `entrypoint_regime.sh` 的 `run_regime_gate.py` 行加上对应 flag。

## 7. 排错

读最近一次执行的容器日志(真实 traceback 在这里,CLI 只报 "execution failed"):

```bash
# 把 JOB 换成 concept-graph-daily 或 regime-daily
JOB=regime-daily
gcloud run jobs executions list --job $JOB --region us-central1 --limit 5
gcloud logging read "resource.type=\"cloud_run_job\" resource.labels.job_name=\"$JOB\"" \
  --project mystockproject-431701 --freshness=3h --order=desc --limit=200 \
  --format='value(textPayload)' | grep -iE "error|traceback|HTTPError|for url|Permission" | head
```

已知坑:

| 现象 | 原因 / 解法 |
|---|---|
| `ModuleNotFoundError: db_dtypes` / `Please install 'db-dtypes'` | `to_dataframe` 缺依赖 → 已在 `[regime]` 加 `db-dtypes`+`pyarrow`,重建镜像 |
| `403 ... bigquery.readsessions.create` | Storage Read API 权限缺 → 给 SA 加 `roles/bigquery.user` |
| `402 ... Special Endpoint ... symbol` (如 `BRK.B`) | FMP 用连字符(`BRK-B`);`fmp.get_financial_statements` 已做 `.`→`-` 归一化 |
| `No module named 'google.genai'` | `[regime]` 已显式声明 `google-genai`,重建镜像 |
| 跑太久,逼近 09:30 开盘 | 提 `--max-workers`、L1 用 flash、降 `--max-news-tickers`;`--task-timeout` 已 7200s |
| OOM(全量概念图) | `gcloud run jobs update concept-graph-daily --memory 8Gi` |
| `regime-daily` 报 `concept-graph snapshot missing on GCS: gs://.../concept_graph/<session>/...` | `concept-graph-daily` 当天没跑成功或还没跑完 → 先确认它本会话已成功(`gcloud run jobs executions list --job concept-graph-daily`),或手动 `gcloud run jobs execute concept-graph-daily --wait` 后再重跑 regime;两个 Scheduler 间隔太近也会撞上,适当拉大 regime 的触发时间 |
