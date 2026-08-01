from langchain_core.documents import Document
from typing import Dict, List, Optional
from pathlib import Path
import hashlib
import logging
import re

logger = logging.getLogger(__name__)


class DataIngestion:
    """使用 unstructured 库加载并预处理文档。"""

    # PDF OCR 语言 —— 默认简体中文 + 英文
    DEFAULT_LANGUAGES = ["chi_sim", "eng"]

    # 支持的文件后缀
    _SUFFIX_MAP = {
        ".pdf":  "pdf",
        ".txt":  "text",
        ".md":   "text",
        ".log":  "text",
        ".html": "html",
        ".htm":  "html",
        ".csv":  "csv",
        ".docx": "docx",
    }

    # 排除的文件名（大小写不敏感）—— 说明文档、目录文件等不应向量化
    _EXCLUDED_FILENAMES = {
        "readme.md",
        "readme.txt",
        "readme",
        "changelog.md",
        "changelog.txt",
        "license",
        "license.md",
        "license.txt",
    }

    # ------------------------------------------------------------------
    # 文件级工具（哈希 / 增量差异对比）
    # ------------------------------------------------------------------

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """返回文件的 SHA-256 十六进制摘要。"""
        sha = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @classmethod
    def scan_files(cls, paths: List[str]) -> Dict[str, str]:
        """
        递归遍历路径，为每个支持的文件返回 {绝对路径: sha256}。

        调用方用此方法在两次索引构建之间检测新增与变更。
        """
        result: Dict[str, str] = {}
        for base_path in paths:
            base = Path(base_path)
            files = base.rglob("*") if base.is_dir() else [base]
            for fp in files:
                if fp.is_dir():
                    continue
                if fp.suffix.lower() not in cls._SUFFIX_MAP:
                    continue
                try:
                    result[str(fp.resolve())] = cls.compute_file_hash(fp)
                except Exception as e:
                    logger.warning("[哈希跳过] %s → %s", fp, e)
        return result

    @classmethod
    def diff_files(
        cls,
        paths: List[str],
        known_hashes: Dict[str, str],
    ) -> Dict[str, list]:
        """
        将当前磁盘文件与已知哈希快照做对比。

        Returns:
            {
                "new":            [不在已知快照中的路径],
                "changed":        [哈希发生变化的路径],
                "removed":        [在已知快照中但磁盘上已不存在的路径],
                "current_hashes": {path: sha256}  # 完整快照
            }
        """
        current = cls.scan_files(paths)
        new_files = [p for p in current if p not in known_hashes]
        changed_files = [p for p in current if p in known_hashes and current[p] != known_hashes[p]]
        removed_files = [p for p in known_hashes if p not in current]

        return {
            "new": new_files,
            "changed": changed_files,
            "removed": removed_files,
            "current_hashes": current,
        }

    # ------------------------------------------------------------------
    # 能力检测（hi_res vs auto）
    # ------------------------------------------------------------------

    @staticmethod
    def check_pdf_capabilities() -> Dict[str, bool]:
        """
        检测当前环境可用的 PDF 处理能力。

        返回示例::

            {
                "hi_res_ready":   True/False,   # 满足 hi_res 的全部条件
                "ocr_available":  True/False,   # tesseract + chi_sim 可用
                "gpu_available":  True/False,   # CUDA GPU 可见
                "pdf2image_ok":   True/False,   # poppler 已安装
            }

        调用方可据此决定请求 hi_res 还是降级到 auto。
        """
        caps = {
            "hi_res_ready":  False,
            "ocr_available": False,
            "gpu_available": False,
            "pdf2image_ok":  False,
        }

        # 1. GPU 检测
        try:
            import torch
            caps["gpu_available"] = torch.cuda.is_available()
        except Exception:
            pass

        # 2. Tesseract OCR 检测
        try:
            import pytesseract
            langs = pytesseract.get_languages()
            has_chi = any("chi" in l for l in langs)
            caps["ocr_available"] = bool(langs) and has_chi
        except Exception:
            pass

        # 3. pdf2image / poppler 检测
        try:
            from pdf2image import convert_from_bytes
            caps["pdf2image_ok"] = True
        except Exception:
            pass

        # 4. detectron2 检测（最重的依赖）
        try:
            import detectron2  # noqa: F401
            caps["hi_res_ready"] = True  # 前面各步均已通过
        except Exception:
            pass

        return caps

    @classmethod
    def resolve_pdf_strategy(
        cls,
        requested: str,
        verbose: bool = True,
    ) -> str:
        """
        将请求的 PDF 策略解析为实际可用的策略。

        若请求 ``"hi_res"`` 但必要组件缺失，则输出诊断信息并降级返回 ``"auto"``。
        """
        if requested != "hi_res":
            return requested

        caps = cls.check_pdf_capabilities()

        if caps["hi_res_ready"]:
            if not caps["gpu_available"] and verbose:
                logger.warning(
                    "请求了 hi_res 但未检测到 GPU，CPU 处理将非常缓慢——建议使用 'auto'。"
                )
            return "hi_res"

        # 诊断缺失组件
        missing = []
        if not caps["pdf2image_ok"]:
            missing.append("pdf2image / poppler（系统包）")
        if not caps["ocr_available"]:
            missing.append("tesseract OCR + chi_sim 语言包")
        missing.append("detectron2")

        if verbose:
            logger.warning(
                "请求了 hi_res 但缺少依赖: %s，降级为 strategy='auto'。",
                "、".join(missing)
            )
            logger.info(
                "安装系统依赖: tesseract-ocr、poppler；"
                "然后 pip install 'unstructured[pdf]'"
            )

        return "auto"

    # ------------------------------------------------------------------
    # 扫描件 PDF 检测
    # ------------------------------------------------------------------

    @staticmethod
    def _is_scanned_pdf(file_path: str, sample_pages: int = 5, min_chars_per_page: int = 40) -> bool:
        """
        快速检测 PDF 是否为扫描件。

        使用 pdfminer 提取第 0 到 sample_pages-1 页（跳过第 0 页封面），
        若非空页面的平均字符数低于 min_chars_per_page，则判定为无文本层的图片型 PDF。
        """
        try:
            from pdfminer.high_level import extract_text
        except ImportError:
            return False

        try:
            text = extract_text(file_path, page_numbers=list(range(sample_pages)))
        except Exception:
            return False

        if not text or not text.strip():
            return True

        pages = [p.strip() for p in text.split("\f") if p.strip()]
        if not pages:
            return True

        # 跳过封面页，忽略字符数 < 20 的空白/纯图片页
        content_pages = [p for i, p in enumerate(pages) if i > 0 and len(p) >= 20]

        if not content_pages:
            # 所有页面均为空白或封面 —— 大概率是扫描件
            return True

        total_chars = sum(len(p) for p in content_pages)
        return (total_chars / len(content_pages)) < min_chars_per_page

    @staticmethod
    def load_pdf_ocr(
        file_path: str,
        languages: Optional[List[str]] = None,
        dpi: int = 300,
    ):
        """
        使用 pdf2image + pytesseract 对扫描件 PDF 进行 OCR。

        当 unstructured 的 hi_res 策略不可用（缺少 detectron2）时使用。
        返回 list of dict: [{"text": str, "page_number": int}, ...]
        """
        import os as _os
        from pdf2image import convert_from_path
        import pytesseract

        if languages is None:
            languages = DataIngestion.DEFAULT_LANGUAGES

        tessdata_dir = _os.environ.get("TESSDATA_PREFIX", "")
        if not tessdata_dir or not _os.path.isdir(tessdata_dir):
            fallback = _os.path.expanduser("~/tessdata")
            if _os.path.isdir(fallback):
                tessdata_dir = fallback

        try:
            from pypdf import PdfReader
            total_pages = len(PdfReader(file_path).pages)
        except Exception:
            total_pages = 0

        tesseract_lang = "+".join(languages)
        results: list = []

        for start in range(0, total_pages, 10):
            end = min(start + 10, total_pages)
            first = start + 1  # pdf2image uses 1-based page numbers

            images = convert_from_path(
                file_path, first_page=first, last_page=end, dpi=dpi
            )

            for i, img in enumerate(images):
                page_num = first + i

                # Set TESSDATA_PREFIX for this process so pytesseract finds chi_sim
                if tessdata_dir:
                    _os.environ.setdefault("TESSDATA_PREFIX", tessdata_dir)

                text = pytesseract.image_to_string(
                    img, lang=tesseract_lang
                )
                text = DataIngestion.preprocess_text(text)
                text = DataIngestion.ocr_postprocess(text)

                if text:
                    results.append({"text": text, "page_number": page_num})

            if len(images) == 0:
                logger.debug("pdf2image 返回空结果，页码 %d-%d", first, end)

        return results

    @staticmethod
    def load_pdf(
        file_path: str,
        strategy: str = "auto",
        languages: Optional[List[str]] = None,
    ):
        """
        加载 PDF，大文件自动按页拆分为批次处理以限制内存。

        批次大小因策略而异:
          - ``"hi_res"`` → 20 页/批（图片渲染内存开销大）
          - ``"auto"``  → 200 页/批（文本提取内存开销小）
        """
        from unstructured.partition.pdf import partition_pdf

        # 获取总页数
        try:
            from pypdf import PdfReader, PdfWriter
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
        except Exception:
            total_pages = 0

        batch_size = 20 if strategy == "hi_res" else 200

        if total_pages <= batch_size:
            return partition_pdf(
                filename=file_path,
                strategy=strategy,
                languages=languages or DataIngestion.DEFAULT_LANGUAGES,
                include_page_breaks=True,
            )

        # 大 PDF：通过临时文件按页批次处理
        import tempfile
        import os

        logger.info(
            "检测到大 PDF（%d 页）—— 按 %d 页/批处理（策略=%s）",
            total_pages, batch_size, strategy
        )

        all_elements = []
        for start in range(0, total_pages, batch_size):
            end = min(start + batch_size, total_pages)

            writer = PdfWriter()
            for i in range(start, end):
                writer.add_page(reader.pages[i])

            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            try:
                with open(tmp_path, "wb") as f:
                    writer.write(f)

                elements = partition_pdf(
                    filename=tmp_path,
                    strategy=strategy,
                    languages=languages or DataIngestion.DEFAULT_LANGUAGES,
                    include_page_breaks=True,
                )

                # 修正页码偏移：临时 PDF 中页码从 1 开始
                page_offset = start
                for el in elements:
                    try:
                        pn = el.metadata.page_number
                        if pn is not None:
                            el.metadata.page_number = int(pn) + page_offset
                    except Exception:
                        pass

                all_elements.extend(elements)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        return all_elements

    @staticmethod
    def load_text(file_path: str):
        from unstructured.partition.text import partition_text
        return partition_text(filename=file_path)

    @staticmethod
    def load_html(file_path: str):
        from unstructured.partition.html import partition_html
        return partition_html(filename=file_path, include_page_breaks=True)

    @staticmethod
    def load_csv(file_path: str):
        from unstructured.partition.csv import partition_csv
        return partition_csv(filename=file_path)

    @staticmethod
    def load_docx(file_path: str):
        try:
            from unstructured.partition.docx import partition_docx
        except ImportError:
            raise ImportError(
                "需要 python-docx 才能处理 .docx 文件。"
                "请执行: pip install 'unstructured[docx]'"
            )
        return partition_docx(filename=file_path, include_page_breaks=True)

    @staticmethod
    def load_auto(file_path: str):
        """使用 unstructured 通用 partition() 自动检测并解析。"""
        from unstructured.partition.auto import partition
        return partition(filename=file_path, include_page_breaks=True)

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        最小化清洗，保留段落结构供分块使用。

        仅压缩非换行空白并折叠过多空行。
        这是有意为之：下游 recursive_chunking 将 \\n\\n 作为首要分隔符，
        因此必须保留段落边界。
        """
        if not text:
            return ""

        # 仅压缩水平空白（空格、制表符、不间断空格——不包含换行符）
        text = re.sub(r"[^\S\n]+", " ", text)

        # 3 个以上连续换行折叠为恰好 2 个（一个空行）
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def ocr_postprocess(text: str) -> str:
        """
        清理中文 OCR 输出中的常见伪影:

        1. CJK 字符间被插入的空格（"库 存 管 理" → "库存管理"）
        2. 句中被截断的行——将不以句末标点结尾的行与下一行合并
        3. 全角引号规范化
        """
        if not text:
            return text

        # 1. 移除 CJK 字符间的空格。
        #    使用前瞻匹配第二个 CJK 字符，避免在 "CJK-空格-CJK-空格-CJK"
        #    这种 OCR 常见交替模式中留下未匹配的空隙。
        _CJK = (
            "\u4e00-\u9fff"   # CJK 统一表意文字
            "\u3400-\u4dbf"   # CJK 扩展 A
            "\uf900-\ufaff"   # CJK 兼容表意文字
            "\u3000-\u303f"   # CJK 符号与标点
            "\uff00-\uffef"   # 半角与全角形式
        )
        text = re.sub(rf"([{_CJK}])\s+(?=[{_CJK}])", r"\1", text)

        # 2. 合并断行：若一行不以句末标点结尾，则与下一行合并
        _SENTENCE_END = r"。！？…～）\)」』》】\"\'”’"
        lines = text.split("\n")
        merged: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                merged.append(line)
                continue
            if merged and merged[-1].strip() and not re.search(
                rf"[{_SENTENCE_END}]$", merged[-1].rstrip()
            ):
                # 前一行未结束 —— 合并，不加换行
                merged[-1] = merged[-1].rstrip() + stripped
            else:
                merged.append(line)
        text = "\n".join(merged)

        # 3. 规范中文 OCR 常见混淆
        replacements = {
            "　": " ",      # 全角空格 → 半角空格
            "，": "，",     # 全角逗号保留
            "‘": "'",      # 左单引号
            "’": "'",      # 右单引号
            "“": "\"",     # 左双引号
            "”": "\"",     # 右双引号
        }
        # 仅对引号做规范化（不动 CJK 标点）
        for old, new in replacements.items():
            if old != "，":
                text = text.replace(old, new)

        return text.strip()

    @staticmethod
    def normalize_metadata(element, file_path: Path, file_hash: str = "", pdf_strategy: str = "") -> dict:
        """
        将 unstructured Element 的元数据转换为我们的规范格式。

        相比旧版 LangChain 加载器新增的字段:
          - element_type  → Title | NarrativeText | Table | ListItem | …
          - text_as_html  → HTML 表格标记（仅 Table 元素非空）
          - parent_id     → 关联子元素与其父元素（标题等）
          - file_hash     → 源文件 SHA-256（用于增量索引）
          - ocr_strategy  → 使用的 PDF 策略（auto / hi_res），用于追溯 OCR 质量
        """
        try:
            meta = element.metadata.to_dict()
        except AttributeError:
            meta = {}
            if hasattr(element, "metadata") and hasattr(element.metadata, "__dict__"):
                meta = vars(element.metadata).copy()

        domain = file_path.parent.name.lower() if file_path.parent.name else "root"

        return {
            "source":        str(file_path),
            "filename":      file_path.name,
            "page":          meta.get("page_number", "na"),
            "domain":        domain,
            "element_type":  str(meta.get("category", "") or ""),
            "text_as_html":  meta.get("text_as_html", "") or "",
            "parent_id":     str(meta.get("parent_id", "") or ""),
            "file_hash":     file_hash,
            "ocr_strategy":  pdf_strategy,
        }

    # ------------------------------------------------------------------
    # 主入口（签名向下兼容）
    # ------------------------------------------------------------------

    @classmethod
    def ingest(
        cls,
        paths: List[str],
        pdf_strategy: str = "auto_detect",
        languages: Optional[List[str]] = None,
        strategy_overrides: Optional[Dict[str, str]] = None,
        skip_hashes: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        从文件路径或目录加载文档。

        Args:
            paths:        文件或目录路径，递归扫描。
            pdf_strategy: 默认 PDF 策略:
                          ``"auto"``        — pdfminer（仅适用于有文本层的 PDF）
                          ``"hi_res"``      — OCR + 布局分析（扫描件 PDF）
                          ``"auto_detect"`` — 自动检测: 扫描件用 hi_res，文本层用 auto（默认）
            languages:    PDF OCR 语言。默认为 ``["chi_sim", "eng"]``。
            strategy_overrides:
                可选的 {路径前缀: PDF 策略} 映射，覆盖匹配文件的 pdf_strategy。
            skip_hashes:
                可选的 {绝对路径: sha256} 映射，来自上一次构建。
                磁盘哈希匹配的文件将被跳过（增量导入）。

        Returns:
            带富元数据的 LangChain Document 列表。
        """
        if languages is None:
            languages = cls.DEFAULT_LANGUAGES

        # 若依赖缺失，将 hi_res 降级为 auto
        _default_strategy = cls.resolve_pdf_strategy(pdf_strategy)
        _auto_strategy = cls.resolve_pdf_strategy("auto")

        all_docs: List[Document] = []
        skipped_count = 0
        errors: List[tuple] = []  # (文件路径, 错误类型, 错误信息)

        for base_path in paths:
            base = Path(base_path)
            files = base.rglob("*") if base.is_dir() else [base]

            for file_path in files:
                if file_path.is_dir():
                    continue

                suffix = file_path.suffix.lower()
                if suffix not in cls._SUFFIX_MAP:
                    continue

                # 跳过排除列表中的文件（README、CHANGELOG、LICENSE 等）
                if file_path.name.lower() in cls._EXCLUDED_FILENAMES:
                    logger.debug("跳过排除文件: %s", file_path)
                    continue

                abs_path = str(file_path.resolve())

                # 计算文件哈希（用于增量跳过 + 元数据）
                try:
                    file_hash = cls.compute_file_hash(file_path)
                except Exception as e:
                    logger.warning("[跳过] %s → 无法计算哈希: %s", file_path, e)
                    continue

                # 增量：跳过未变更的文件
                if skip_hashes and abs_path in skip_hashes:
                    if skip_hashes[abs_path] == file_hash:
                        skipped_count += 1
                        continue

                # 按文件解析策略
                strategy = cls._resolve_strategy(
                    file_path, _default_strategy, strategy_overrides
                )

                # auto_detect：判断该 PDF 是否需要 OCR
                effective_strategy = strategy
                ocr_fallback = False
                if suffix == ".pdf" and strategy == "auto_detect":
                    if cls._is_scanned_pdf(abs_path):
                        effective_strategy = "hi_res"
                        # 在使用点重新检查 hi_res 可用性
                        effective_strategy = cls.resolve_pdf_strategy(effective_strategy)
                        # hi_res 不可用（缺少 detectron2）→ 回退到本地 OCR
                        if effective_strategy == "auto":
                            ocr_fallback = True
                    else:
                        effective_strategy = _auto_strategy

                try:
                    if ocr_fallback:
                        # 使用 pdf2image + pytesseract 直接 OCR，不依赖 detectron2
                        logger.info("使用本地 OCR 处理扫描件: %s", file_path.name)
                        page_results = cls.load_pdf_ocr(abs_path, languages)
                        for pr in page_results:
                            all_docs.append(Document(
                                page_content=pr["text"],
                                metadata={
                                    "source": abs_path,
                                    "filename": file_path.name,
                                    "page": pr["page_number"],
                                    "domain": file_path.parent.name.lower() if file_path.parent.name else "root",
                                    "element_type": "NarrativeText",
                                    "text_as_html": "",
                                    "parent_id": "",
                                    "file_hash": file_hash,
                                    "ocr_strategy": "tesseract",
                                },
                            ))
                    else:
                        elements = cls._load_file(
                            file_path, suffix, effective_strategy, languages
                        )

                        for element in elements:
                            text = element.text or ""
                            text = cls.preprocess_text(text)

                            # hi_res 文档的 OCR 后处理
                            if suffix == ".pdf" and effective_strategy == "hi_res":
                                text = cls.ocr_postprocess(text)

                            if not text:
                                continue

                            all_docs.append(Document(
                                page_content=text,
                                metadata=cls.normalize_metadata(
                                    element, file_path, file_hash, effective_strategy
                                ),
                            ))

                except ImportError as e:
                    errors.append((str(file_path), "缺少依赖", str(e)))
                except Exception as e:
                    errors.append((str(file_path), type(e).__name__, str(e)))

        if skipped_count:
            logger.info("跳过了 %d 个未变更的文件", skipped_count)
        logger.info("已加载文档: %d", len(all_docs))

        if errors:
            logger.warning("%d 个文件加载失败:", len(errors))
            for path, err_type, msg in errors:
                logger.warning("  %s [%s] %s", path, err_type, msg)

        if not all_docs:
            logger.warning("未找到任何文档，索引将为空。")

        return all_docs

    @classmethod
    def _resolve_strategy(
        cls,
        file_path: Path,
        default_strategy: str,
        overrides: Optional[Dict[str, str]],
    ) -> str:
        """通过前缀匹配策略覆盖，为指定文件选择 PDF 策略。"""
        if overrides:
            abs_path = str(file_path.resolve())
            for prefix, strategy in overrides.items():
                if abs_path.startswith(str(Path(prefix).resolve())):
                    return strategy
        return default_strategy

    @classmethod
    def _load_file(cls, file_path: Path, suffix: str, strategy: str, languages: List[str]):
        """将单个文件路由到对应的 unstructured 解析器。"""
        if suffix == ".pdf":
            return cls.load_pdf(str(file_path), strategy=strategy, languages=languages)
        elif suffix in (".txt", ".md", ".log"):
            return cls.load_text(str(file_path))
        elif suffix in (".html", ".htm"):
            return cls.load_html(str(file_path))
        elif suffix == ".csv":
            return cls.load_csv(str(file_path))
        elif suffix == ".docx":
            return cls.load_docx(str(file_path))
        return []
