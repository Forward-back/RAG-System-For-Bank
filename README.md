# RAG System For Bank — 银行业务智能问答系统

基于 RAG（检索增强生成）架构的企业级银行业务文档智能问答系统，支持多策略分块、混合检索、重排序、上下文压缩、Text-to-SQL、三级安全校验及联网搜索兜底

## 架构概览

```
用户查询 → 查询分类 → 查询改写 → 查询扩展 → 混合检索(稠密+BM25)
    → 重排序 → 上下文压缩 → LLM生成(带引用) → L1/L2/L3安全校验 → 答案
```

### 核心模块

| 模块 | 说明 |
|------|------|
| **Query Classification** | 风险等级 / 文档类型 / 计算需求分类，高风险问题拒绝回答 |
| **Query Rewriting** | 歧义查询改写，提升检索命中率 |
| **Query Expansion** | 模板/LM 多路查询扩展 |
| **Hybrid Retrieval** | ChromaDB 稠密检索 + BM25 稀疏检索，支持 RRF/加权求和融合 |
| **Re-Ranking** | Cross-Encoder 重排序（BGE/BCE 可切换） |
| **Context Compression** | LLM 驱动的上下文去冗压缩 |
| **Text-to-SQL** | 结构化数据库查询，自动生成 SQL 并执行 |
| **Generation** | DeepSeek LLM 生成带源引用的答案 |
| **Safety (L1→L2→L3)** | NumericGuard（规则）→ NLI Verifier（交叉编码器）→ FactChecker（LLM） |
| **Web Search Fallback** | 知识库检索质量不足时自动联网搜索兜底 |
| **Query Cache** | 语义级查询缓存，FAQ 场景命中率高 |

## 快速开始

### 环境要求

- Python 3.11+
- Tesseract OCR（用于 PDF OCR，需安装中文语言包）
- Poppler（用于 PDF 图片提取）
- MySQL（Text-to-SQL 功能需要）

### 本地开发

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env  # 编辑 .env 填入 API Key

# 4. 准备数据
mkdir data && cp /path/to/your/pdfs data/

# 5. 启动 API 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. 启动 UI（可选）
pip install -r requirements.ui.txt
streamlit run UI/app.py --server.port 8501
```

### Docker 部署

```bash
docker compose up -d
```

服务启动后：
- API: `http://localhost:8000`
- API 文档: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

## 配置

通过 `.env` 文件或环境变量配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | - | DeepSeek API Key（必填） |
| `DEEPSEEK_MODEL` | `deepseek-chat` | LLM 模型 |
| `CHUNKING_MODE` | `recursive` | 分块策略：`recursive` / `semantic` / `element_aware` / `hierarchical` |
| `TOP_K` | `5` | 检索返回文档数 |
| `ENABLE_RERANK` | `true` | 启用重排序 |
| `ENABLE_COMPRESSION` | `true` | 启用上下文压缩 |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 重排序模型 |
| `ENABLE_WEB_SEARCH` | `false` | 启用联网搜索兜底 |
| `ENABLE_QUERY_CACHE` | `true` | 启用查询缓存 |
| `PDF_STRATEGY` | `auto_detect` | PDF 解析策略 |
| `API_KEY` | - | API 鉴权密钥（不设则跳过鉴权） |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/query` | 问答查询（限流 30次/分钟） |
| `POST` | `/rebuild-index` | 重建索引（限流 2次/分钟） |

### 查询请求示例

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "试用期员工的年假政策是什么？", "enable_web_search": false}'
```

## 项目结构

```
├── app/                    # FastAPI 应用
│   ├── main.py             # API 入口、中间件、启动流程
│   ├── schemas.py          # 请求/响应模型
│   └── logger.py           # 日志配置
├── src/
│   ├── rag_pipeline.py     # RAG 主流程编排
│   ├── generation/         # LLM 生成（DeepSeek）
│   ├── ingestion/          # 文档摄取、分块、表格处理、层级解析
│   ├── retrieval/          # 向量存储、混合检索、重排序、查询变换、上下文压缩
│   ├── query/              # 查询分类、改写、Text-to-SQL
│   ├── safety/             # 数值守卫(L1)、NLI校验(L2)、事实核查(L3)
│   └── infra/              # 缓存、追踪、数据库、网络搜索、模型注册
├── UI/                     # Streamlit 前端
├── evaluation/             # 评测脚本与数据集
├── scripts/                # 工具脚本
├── data/                   # 文档数据（gitignore）
├── chroma_db/              # 向量数据库持久化（gitignore）
├── Dockerfile.api          # API 服务镜像
├── Dockerfile.ui           # UI 服务镜像
└── docker-compose.yml      # 多服务编排
```

## 评测

```bash
cd evaluation

# 基线评测
python eval_baseline.py

# 分块策略对比
python eval_chunking.py

# 重排序策略对比
python eval_rerank.py

# Top-K 消融实验
python eval_topk.py
```

## License

MIT
