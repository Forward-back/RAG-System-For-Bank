# Ingestion 板块

## 概述

Ingestion 板块是 RAG pipeline 的数据基石，负责**文档加载 → 预处理 → 分块 → 表格提取**全流程。
产出是带富元数据的 LangChain `Document` 列表，供下游 EmbeddingStore 向量化和 HybridRetriever 检索。

四个核心组件：

| 组件 | 文件 | 职责 |
|------|------|------|
| `DataIngestion` | `documents_ingestion.py` | 文件扫描、哈希增量检测、PDF/文本/CSV/DOCX 加载、OCR、预处理 |
| `Chunking` | `chunking.py` | 4 种分块策略，将文档切分为适合嵌入的片段 |
| `TableProcessor` | `table_processor.py` | 表格关键词提取（post-chunking），生成 BM25 友好的索引文本 |
| `HierarchyParser` | `hierarchy_parser.py` | 中文法规层级解析 + 检索时上下文扩展 |

---

## 数据流

```
文件系统                          向量库
   │                               ▲
   ▼                               │
DataIngestion.scan_files()          │
   │  {path: sha256}                │
   ▼                               │
DataIngestion.diff_files()          │
   │  {new, changed, removed}       │
   ▼                               │
DataIngestion.ingest()              │
   │  List[Document]                │
   │  - page_content: 原始文本       │
   │  - metadata: source, page,      │
   │    element_type, text_as_html,  │
   │    file_hash, ocr_strategy     │
   ▼                               │
Chunking.{mode}_chunking()          │
   │  List[Document]                │
   │  - first_page / last_page      │
   │  - is_table (element_aware)    │
   │  - hierarchy_path (hierarchical)│
   ▼                               │
TableProcessor.process()            │
   │  仅处理 is_table 或 CSV 文档    │
   │  - page_content → 关键词        │
   │  - text_as_html → 原始 HTML 保留│
   ▼                               │
EmbeddingStore.create_or_load_db() ─┘
```

---

## 1. DataIngestion — 文档加载

### 1.1 支持格式

| 格式 | 后缀 | 解析引擎 |
|------|------|---------|
| PDF（文本层） | `.pdf` | unstructured `partition_pdf` strategy=`"auto"`（pdfminer） |
| PDF（扫描件） | `.pdf` | unstructured `partition_pdf` strategy=`"hi_res"`（OCR + 布局检测） |
| 纯文本 | `.txt` `.md` `.log` | unstructured `partition_text` |
| HTML | `.html` `.htm` | unstructured `partition_html` |
| CSV | `.csv` | unstructured `partition_csv` |
| Word | `.docx` | unstructured `partition_docx` |

### 1.2 PDF 策略

| 策略 | 后端 | 适用 | 依赖 |
|------|------|------|------|
| `auto` | pdfminer.six | 有文本层的普通 PDF | 无额外依赖 |
| `hi_res` | OCR + detectron2 布局 | 扫描件、图片型 PDF | tesseract + chi_sim、poppler、detectron2、torch |
| `auto_detect`（默认） | 自动切换 | 混合场景 | 按需 |

`auto_detect` 会采样前 5 页，若平均每页字符数 < 40 则判定为扫描件，自动切换到 `hi_res`。

### 1.3 大 PDF 分批加载

超过批次阈值的 PDF 自动按页拆分为临时文件分批处理，避免 OOM：

| 策略 | 批次大小 | 依据 |
|------|---------|------|
| `auto` | 200 页 | 纯文本提取，内存占用极低 |
| `hi_res` | 20 页 | 每页渲染为图像约 24MB（A4 300DPI），20 页约 480MB + 模型 |

批次处理后自动修正页码偏移。

### 1.4 预处理

- **`preprocess_text()`**：压缩水平空白（保留换行符维护段落边界），折叠 3+ 连续空行
- **`ocr_postprocess()`**（仅 hi_res 模式）：
  1. 移除 CJK 字符间被 OCR 插入的空格
  2. 合并句中被截断的行
  3. 统一全角引号

### 1.5 增量索引

通过 SHA-256 文件哈希实现：

1. `scan_files()` → `{path: sha256}`
2. `diff_files()` → 对比上次快照，返回 `{new, changed, removed, current_hashes}`
3. `ingest()` 通过 `skip_hashes` 跳过未变更文件
4. `build_index()` 对 changed/removed 文件执行 delete-by-source 清理旧 chunk

详细流程见 `rag_pipeline.py` 的 `build_index()` 方法。

### 1.6 错误处理

每个文件独立 try/except，解析失败不中断整体流程。错误信息按 `(文件路径, 错误类型, 错误信息)` 三元组收集，在结尾统一输出摘要。

### 1.7 元数据规范

每个 Document 的 metadata 包含：

| 字段 | 类型 | 说明 | 来源 |
|------|------|------|------|
| `source` | str | 文件绝对路径 | 文件系统 |
| `filename` | str | 文件名 | 文件系统 |
| `page` | int/str | 页码或页码范围 | unstructured |
| `domain` | str | 领域分类 | **父目录名**（一级扁平结构） |
| `element_type` | str | Title / NarrativeText / Table / ListItem / … | unstructured |
| `text_as_html` | str | 原始 HTML 表格标记 | unstructured（仅 Table 元素） |
| `parent_id` | str | 父元素 ID | unstructured |
| `file_hash` | str | 文件 SHA-256 | `compute_file_hash()` |
| `ocr_strategy` | str | 使用的 PDF 策略 | `ingest()` |

---

## 2. Chunking — 分块策略

### 2.1 策略对比

| 策略 | 模式值 | 速度 | 表原子性 | 结构感知 | 推荐场景 |
|------|--------|------|----------|----------|----------|
| `element_aware` | `"element_aware"` | 快 | 原生支持 | 元素类型 | **生产默认** |
| `hierarchical` | `"hierarchical"` | 快 | 原生支持 | 法规层级 | 法规/条例 |
| `recursive` | `"recursive"` | 快 | 不支持 | 分隔符 | 回退/测试 |
| `semantic` | `"semantic"` | 极慢 | 不支持 | 语义相似度 | 实验性 |

通过环境变量 `CHUNKING_MODE` 配置，默认 `element_aware`。

### 2.2 element_aware（生产默认）

利用 unstructured 元素类型标签指导切分：

- **Title**：作为 section 标题，前缀到后续 chunk
- **Table**：原子保留，不切割，标记 `is_table: True`
- **NarrativeText / ListItem**：累积到 `chunk_size` 后 flush
- 超大段落回退到 recursive 进一步切分

适用：所有通过 unstructured 解析的文档，通用性最强。

### 2.3 hierarchical

解析中文法规的六层层级结构：

| 层级 | 名称 | 模式 | 示例 |
|------|------|------|------|
| 1 | 章 | `第[数字]+章` | 第一章 总则 |
| 2 | 节 | `第[数字]+节` | 第一节 适用范围 |
| 3 | 条 | `第[数字]+条` | 第一条 |
| 4 | 项 | `（[数字]+）` | （一） |
| 5 | 目 | `数字.` 或 `数字、` | 1. |
| 6 | 细目 | `①-⑳` | ① |

每个 chunk 携带层级元数据：`hierarchy_path`、`hierarchy_level`、`hierarchy_heading`、`parent_heading`、`article_heading`。

检索时通过 `expand_context()` 自动补全父级和同级 chunk，使用预建的 O(1) hashmap 索引。

### 2.4 recursive

按分隔符优先级递归切割：

```
\n\n → \n → 。→ ？→ ！→ ；→ ，→ . → ! → ? → ; → , → 空格 → 字符
```

### 2.5 semantic（实验性）

逐句 embedding（SentenceTransformer），在低相似度点切分。

**生产不推荐**：每次 `build_index` 需重新 embedding 全部句子，无缓存，大批量文档下耗时极高。

### 2.6 页码规范

每个 chunk 携带 `first_page` 和 `last_page`，多页 chunk 的 `page` 字段为 `"3-5"` 格式。

---

## 3. TableProcessor — 表格关键词提取

### 设计原则

表格采用 **C+E 方案**（Parent-Child 双存 + 双通道检索）：

- **`page_content`**：规则生成的关键词列表，用于 BM25 关键词匹配检索
- **`text_as_html`**：原始 HTML 表格结构，用于 LLM 生成回答

TableProcessor **在 chunking 之后**运行，仅处理 `is_table: True` 或 CSV 来源的文档。

### 关键词格式

```
表格列: 产品名称 | 年利率 | 期限
活期存款 0.35% 无固定 三个月定存 1.35% 3个月 六个月定存 1.55% 6个月 ...
```

### 生成器集成

`RAGGenerator` 在构建 LLM 上下文时，对 `is_table` chunk 自动使用 `text_as_html` 而非 `page_content`，
确保 LLM 看到的是原始表格结构，而非关键词列表。

---

## 4. HierarchyParser — 法规层级解析

### 解析

`HierarchyParser.parse(text)` 使用栈式解析器将法规文档拆分为 `HierarchySegment` 列表，
每个 segment 带有完整的祖先路径。

### 上下文扩展

`expand_context(doc, all_docs, direction, index)` 在检索时补全 chunk 的上下文：

| direction | 行为 |
|-----------|------|
| `"parent"` | 返回父级 chunk |
| `"children"` | 返回下一级子 chunk |
| `"siblings"` | 返回同级 chunk |
| `"all"` | parent + siblings |

通过 `build_hierarchy_index()` 预建 `{(source, heading): [chunks]}` 和 `{(source, parent_path, level): [chunks]}` 两级 hashmap，检索时 O(1) 查找代替 O(n) 全量扫描。

---

## 5. 配置参考

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHUNKING_MODE` | `element_aware` | 分块策略 |
| `DATA_PATH` | `./data` | 文档目录 |
| `CHROMA_DIR` | `./chroma_db` | 向量库持久化目录 |

### 日志级别

```bash
LOG_LEVEL=INFO     # 生产默认
LOG_LEVEL=DEBUG    # 开发调试（含批次进度）
LOG_LEVEL=WARNING  # 仅异常
```

### data/ 目录组织

```
data/
├── 存款产品/
│   ├── 活期存款说明.pdf
│   └── 定期存款利率表.csv
├── 贷款产品/
│   ├── 个人住房贷款规程.pdf
│   └── 消费贷产品手册.docx
└── ...
```

规则：
- **一级扁平目录**，目录名即 `domain`
- **文本文件必须 UTF-8** 编码，GBK/GB2312 等会被跳过并记录警告
- 详见 `data/README.md`
