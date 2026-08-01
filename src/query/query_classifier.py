"""
查询分类器 —— 基于向量相似度对用户查询进行多标签分类。

核心思路（grill-me 讨论确定的方案 D）：
  1. 维护一个 (查询 → 标签) 的样本库
  2. 对所有样本预计算 embedding
  3. 运行时：embed 用户查询 → 找到 k 个最相似样本 → 加权投票

为什么不选方案 A（LLM 分类）：
  - 每次调用增加 1-2 秒延迟
  - 非确定性，同一查询可能被分到不同类别，导致下游回答不一致
  - 每次调用都有 API 费用

为什么不选方案 B（微调 BERT）：
  - 冷启动需要 500+ 条标注数据；本模块本身就是标注数据的采集工具
  - 当积累足够的线上标注查询后，样本库可以直接作为微调模型的训练集

依赖：sentence-transformers（与项目现有的 embedding 基础设施一致），无需 GPU。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from src.infra.model_registry import get_embedding_model
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# 标签体系
# ---------------------------------------------------------------------------
# 每条样本格式：{"query": str, "labels": {"risk_level": ..., "doc_type": ..., "computation_needed": ...}}
#
# risk_level（风险等级）：
#   "safe"     — 事实查询，无财务影响（网点地址、营业时间、操作指引）
#   "caution"  — 涉及费用/利率/额度，需引用原文并附带免责声明
#   "block"    — 涉及投资建议/违约金/金额计算，拒答并转人工
#
# doc_type（文档类型，用于缩小检索范围）：
#   "regulation"  — 规章制度、管理办法
#   "procedure"   — 业务流程、操作指南
#   "product"     — 产品信息（理财、贷款、存款）
#   "faq"         — 常见问答、一般咨询
#   "general"     — 无法归类，全量检索
#
# computation_needed（是否涉及数值计算）：
#   true  — 需要代入公式/计算利率/计算月供等，当前仅返回公式原文，不代为计算
#   false — 纯文本检索


# ---------------------------------------------------------------------------
# 种子样本 —— 银行客服场景
# ---------------------------------------------------------------------------

BANK_SEED_SAMPLES: List[dict] = [
    # —— safe / faq ——
    {"query": "你们营业时间是几点",       "labels": {"risk_level": "safe", "doc_type": "faq", "computation_needed": False}},
    {"query": "周末开门吗",              "labels": {"risk_level": "safe", "doc_type": "faq", "computation_needed": False}},
    {"query": "最近的网点在哪里",         "labels": {"risk_level": "safe", "doc_type": "faq", "computation_needed": False}},
    {"query": "有没有停车位",            "labels": {"risk_level": "safe", "doc_type": "faq", "computation_needed": False}},
    {"query": "你们的客服电话是多少",     "labels": {"risk_level": "safe", "doc_type": "faq", "computation_needed": False}},
    {"query": "可以预约办理业务吗",       "labels": {"risk_level": "safe", "doc_type": "faq", "computation_needed": False}},
    {"query": "办业务需要带什么证件",     "labels": {"risk_level": "safe", "doc_type": "faq", "computation_needed": False}},
    {"query": "身份证没带能办吗",         "labels": {"risk_level": "safe", "doc_type": "faq", "computation_needed": False}},
    {"query": "网点有WiFi吗",            "labels": {"risk_level": "safe", "doc_type": "faq", "computation_needed": False}},
    {"query": "春节放假吗",              "labels": {"risk_level": "safe", "doc_type": "faq", "computation_needed": False}},

    # —— safe / procedure ——
    {"query": "怎么开通手机银行",         "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "网上银行怎么注册",         "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "忘记密码怎么办",           "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "怎么修改预留手机号",       "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "银行卡丢了怎么挂失",       "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "卡被吞了怎么取回",         "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "怎么打印银行流水",         "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "怎么销户",                 "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "换卡需要多久",             "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "怎么绑定微信支付",         "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "对公账户开户流程",         "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "企业网银怎么申请",         "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},
    {"query": "如何更新身份证信息",       "labels": {"risk_level": "safe", "doc_type": "procedure", "computation_needed": False}},

    # —— safe / regulation ——
    {"query": "个人账户分类管理规定",     "labels": {"risk_level": "safe", "doc_type": "regulation", "computation_needed": False}},
    {"query": "大额交易报告标准",         "labels": {"risk_level": "safe", "doc_type": "regulation", "computation_needed": False}},
    {"query": "反洗钱规定是什么",         "labels": {"risk_level": "safe", "doc_type": "regulation", "computation_needed": False}},
    {"query": "个人信息保护政策",         "labels": {"risk_level": "safe", "doc_type": "regulation", "computation_needed": False}},

    # —— caution / product ——
    {"query": "转账手续费是多少",         "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},
    {"query": "跨行转账怎么收费",         "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},
    {"query": "信用卡年费多少",           "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},
    {"query": "信用卡取现手续费",         "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},
    {"query": "短信提醒收费吗",           "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},
    {"query": "账户管理费怎么收",         "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},
    {"query": "小额账户收费标准",         "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},
    {"query": "跨境汇款手续费多少",       "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},
    {"query": "存款利率是多少",           "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},
    {"query": "理财产品有哪些类型",       "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},
    {"query": "贷款额度怎么确定",         "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},
    {"query": "信用卡额度是多少",         "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": False}},

    # —— caution / procedure ——
    {"query": "跨行转账多久到账",         "labels": {"risk_level": "caution", "doc_type": "procedure", "computation_needed": False}},
    {"query": "大额取款需要预约吗",       "labels": {"risk_level": "caution", "doc_type": "procedure", "computation_needed": False}},
    {"query": "理财产品怎么赎回",         "labels": {"risk_level": "caution", "doc_type": "procedure", "computation_needed": False}},
    {"query": "贷款申请需要什么材料",     "labels": {"risk_level": "caution", "doc_type": "procedure", "computation_needed": False}},
    {"query": "信用卡逾期了怎么办",       "labels": {"risk_level": "caution", "doc_type": "procedure", "computation_needed": False}},
    {"query": "房贷提前还款流程",         "labels": {"risk_level": "caution", "doc_type": "procedure", "computation_needed": False}},
    {"query": "怎么查询个人征信",         "labels": {"risk_level": "caution", "doc_type": "procedure", "computation_needed": False}},

    # —— block / product ——
    {"query": "提前还贷要违约金吗",       "labels": {"risk_level": "block", "doc_type": "product", "computation_needed": False}},
    {"query": "定期存款提前取出来利息怎么算", "labels": {"risk_level": "block", "doc_type": "product", "computation_needed": True}},
    {"query": "贷款利率能优惠吗",         "labels": {"risk_level": "block", "doc_type": "product", "computation_needed": False}},
    {"query": "理财产品保本吗",           "labels": {"risk_level": "block", "doc_type": "product", "computation_needed": False}},
    {"query": "理财产品亏损了怎么办",     "labels": {"risk_level": "block", "doc_type": "product", "computation_needed": False}},
    {"query": "哪个理财产品收益高",       "labels": {"risk_level": "block", "doc_type": "product", "computation_needed": False}},
    {"query": "房贷利率最新是多少",       "labels": {"risk_level": "block", "doc_type": "product", "computation_needed": False}},
    {"query": "信用贷款利率怎么算",       "labels": {"risk_level": "block", "doc_type": "product", "computation_needed": True}},
    {"query": "有没有比定期更好的理财方式", "labels": {"risk_level": "block", "doc_type": "product", "computation_needed": False}},
    {"query": "现在适合买房吗",           "labels": {"risk_level": "block", "doc_type": "general", "computation_needed": False}},

    # —— block / procedure ——
    {"query": "信用卡逾期记录怎么消除",   "labels": {"risk_level": "block", "doc_type": "procedure", "computation_needed": False}},
    {"query": "贷款被拒怎么办",           "labels": {"risk_level": "block", "doc_type": "procedure", "computation_needed": False}},
    {"query": "被诈骗了钱能追回来吗",     "labels": {"risk_level": "block", "doc_type": "procedure", "computation_needed": False}},

    # —— 涉及计算 ——
    {"query": "存10万一年利息多少",       "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": True}},
    {"query": "贷款100万30年月供多少",    "labels": {"risk_level": "block",  "doc_type": "product", "computation_needed": True}},
    {"query": "20万理财3个月收益多少",    "labels": {"risk_level": "block",  "doc_type": "product", "computation_needed": True}},
    {"query": "定期到期后自动转存利率怎么算", "labels": {"risk_level": "caution", "doc_type": "product", "computation_needed": True}},
    {"query": "等额本息和等额本金哪个划算", "labels": {"risk_level": "block",  "doc_type": "product", "computation_needed": True}},
]


# ---------------------------------------------------------------------------
# 分类器
# ---------------------------------------------------------------------------

class QueryClassifier:
    """
    通过 embedding 相似度 + 加权投票对用户查询进行多标签分类。

    使用示例::

        clf = QueryClassifier()
        clf.load_samples(BANK_SEED_SAMPLES)   # 或从 JSON 文件加载
        result = clf.classify("转账要手续费吗")
        # → {"risk_level": "caution", "doc_type": "product", "computation_needed": False,
        #     "confidence": 0.87, "nearest_samples": [...]}
    """

    def __init__(self):
        self.model = get_embedding_model()
        self._samples: List[dict] = []
        self._embeddings: Optional[np.ndarray] = None  # shape (样本数, 向量维度)
        self._label_keys = ("risk_level", "doc_type", "computation_needed")

    # ------------------------------------------------------------------
    # 样本管理
    # ------------------------------------------------------------------

    @property
    def sample_count(self) -> int:
        """已加载的样本总数。"""
        return len(self._samples)

    def load_samples(self, samples: List[dict]) -> None:
        """
        加载标注样本并预计算所有 embedding。

        每条样本格式::

            {
                "query":  str,
                "labels": {
                    "risk_level":         "safe" | "caution" | "block",
                    "doc_type":           "regulation" | "procedure" | "product" | "faq" | "general",
                    "computation_needed": true | false
                }
            }

        注意：此方法会全量替换已有样本并重新编码，增量添加请用 add_samples()。
        """
        if not samples:
            raise ValueError("样本列表不能为空")

        queries = [s["query"] for s in samples]
        embeddings = self.model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=len(queries) > 100,
        )

        self._samples = list(samples)
        self._embeddings = np.array(embeddings)
        print(f"[CLASSIFIER] 已加载 {len(self._samples)} 条样本, "
              f"向量维度={self._embeddings.shape[1]}")

    def add_samples(self, samples: List[dict]) -> None:
        """增量添加样本，会触发全量重新编码（适合少量样本的持续积累）。"""
        combined = list(self._samples) + list(samples)
        self.load_samples(combined)

    def save_samples(self, path: str) -> None:
        """
        持久化样本库到磁盘：JSON（样本数据）+ npy（预计算 embedding）。

        后续可通过 load_samples_from_file() 直接恢复，避免每次启动重新编码。
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._samples, f, ensure_ascii=False, indent=2)
        if self._embeddings is not None:
            np.save(str(p.with_suffix(".npy")), self._embeddings)
        print(f"[CLASSIFIER] 已保存 {len(self._samples)} 条样本至 {p}")

    def load_samples_from_file(self, path: str) -> None:
        """
        从之前保存的 JSON + npy 文件恢复样本库。

        若 npy 文件缺失或尺寸不匹配，退化为从 JSON 重新编码。
        """
        p = Path(path)
        npy_path = str(p.with_suffix(".npy"))
        if Path(npy_path).exists():
            self._embeddings = np.load(npy_path)
        with open(p, "r", encoding="utf-8") as f:
            self._samples = json.load(f)
        if self._embeddings is None or len(self._embeddings) != len(self._samples):
            # Embedding 缺失或与样本数不一致 → 重新编码
            self.load_samples(self._samples)
        else:
            self.model.encode(["warmup"], normalize_embeddings=True)  # 预热模型
            print(f"[CLASSIFIER] 从 {p} 加载了 {len(self._samples)} 条样本")

    # ------------------------------------------------------------------
    # 分类
    # ------------------------------------------------------------------

    def classify(
        self,
        query: str,
        k: int = 7,
        min_confidence: float = 0.5,
    ) -> dict:
        """
        对用户查询进行分类，返回多标签预测及置信度。

        步骤：
        1. 将用户查询编码为向量
        2. 计算与所有样本的余弦相似度
        3. 取 top-k，过滤相似度 < 0.3 的噪音邻居
        4. 按相似度加权投票，得出每个标签的预测值

        参数：
            query: 用户原始查询
            k: 投票时参考的近邻数量
            min_confidence: 暂未使用，保留字段

        返回::

            {
                "risk_level":         str,      # "safe" | "caution" | "block"
                "doc_type":           str,      # "regulation" | "procedure" | "product" | "faq" | "general"
                "computation_needed": bool,     # 是否涉及数值计算
                "confidence":         float,    # 0-1，取 top-k 相似度均值
                "vote_detail": {                # 各标签的投票明细
                    "risk_level":         {"safe": 0.6, "caution": 0.3, ...},
                    "doc_type":           {...},
                    "computation_needed": {"true": 0.4, "false": 0.6},
                },
                "nearest_samples": [            # 最近邻样本，便于人工核查
                    {"query": str, "labels": {...}, "similarity": float},
                    ...
                ],
            }
        """
        if self._embeddings is None or len(self._samples) == 0:
            raise RuntimeError("尚未加载样本，请先调用 load_samples()")

        # 编码查询 → 找 top-k
        q_vec = self.model.encode(
            [query], normalize_embeddings=True
        )[0]  # shape (dim,)

        sims = np.dot(self._embeddings, q_vec)        # 余弦相似度（向量已归一化）
        top_idx = np.argsort(sims)[::-1][:k]

        top_sims = sims[top_idx]
        top_samples = [self._samples[i] for i in top_idx]

        # 过滤相似度过低的噪音邻居
        mask = top_sims >= 0.3
        if not mask.any():
            # 无任何样本足够相似 → 返回低置信度默认值
            return {
                "risk_level": "safe",
                "doc_type": "general",
                "computation_needed": False,
                "confidence": 0.0,
                "vote_detail": {},
                "nearest_samples": [],
            }

        top_sims = top_sims[mask]
        top_samples = [top_samples[i] for i in range(len(mask)) if mask[i]]
        k_eff = len(top_samples)

        # 逐标签加权投票
        vote_detail: Dict[str, Dict[str, float]] = {}
        predictions: Dict[str, object] = {}

        for key in self._label_keys:
            votes: Dict[str, float] = defaultdict(float)
            for sample, sim in zip(top_samples, top_sims):
                val = sample["labels"][key]
                val_str = str(val).lower()
                votes[val_str] += float(sim)

            vote_detail[key] = dict(votes)

            # 取票数最高的类别
            if key == "computation_needed":
                true_votes = votes.get("true", 0.0)
                predictions[key] = true_votes > votes.get("false", 0.0)
            else:
                predictions[key] = max(votes, key=lambda v: votes[v])

        # 置信度 = top-k 相似度均值
        confidence = float(np.mean(top_sims))

        # 最近邻样本，供下游追溯决策依据
        nearest = [
            {
                "query": s["query"],
                "labels": s["labels"],
                "similarity": round(float(sim), 4),
            }
            for s, sim in zip(top_samples, top_sims)
        ]

        return {
            "risk_level": predictions["risk_level"],
            "doc_type": predictions["doc_type"],
            "computation_needed": predictions["computation_needed"],
            "confidence": round(confidence, 4),
            "vote_detail": vote_detail,
            "nearest_samples": nearest,
        }

    # ------------------------------------------------------------------
    # 便捷方法：分类 + 路由
    # ------------------------------------------------------------------

    def route(self, query: str, k: int = 7) -> dict:
        """
        分类并直接输出 RAG 管线路由决策，供下游回答策略层直接使用。

        路由逻辑：
          - block   → answer_with_disclaimer（涉及敏感领域，回答并附加免责声明）
          - caution → answer_with_disclaimer（回答 + 免责声明 + 原文引用）
          - safe    → answer（直接回答）
          - 置信度 < 0.4 → answer（不确定时放行，由 L1/L2/L3 安全层把关）

        注意：分类器只提供路由提示，不直接拒答。真正的安全检查由
        NumericGuard (L1) / NLIVerifier (L2) / FactChecker (L3) 在生成后完成。

        返回::

            {
                "risk_level":         str,
                "doc_type":           str,
                "computation_needed": bool,
                "confidence":         float,
                "action":             "answer" | "answer_with_disclaimer",
                "search_filter":      str | None,   # 用于限制检索范围的 doc_type
                "reason":             str,           # 路由原因的中文说明
            }
        """
        result = self.classify(query, k=k)

        risk = result["risk_level"]
        conf = result["confidence"]
        comp = result["computation_needed"]
        dtype = result["doc_type"]

        # 路由决策 — block/caution 都回答，但附加免责声明
        if risk == "block":
            action = "answer_with_disclaimer"
            reason = "涉及投资建议/违约金/利率比较，回答同时附加免责声明"
        elif risk == "caution":
            action = "answer_with_disclaimer"
            reason = "涉及费用/利率/限额，需附加免责声明并引用原文出处"
        else:
            action = "answer"
            reason = "安全查询，直接回答"

        if comp:
            reason += "；该问题涉及数值计算，当前仅提供公式/原文，不代为计算"

        if conf < 0.4:
            # Low confidence → default to safe path; L1/L2/L3 safety layers
            # will catch actual problems post-generation.
            action = "answer"
            reason = f"分类置信度较低({conf:.2f})，默认按安全查询处理"

        return {
            "risk_level": risk,
            "doc_type": dtype,
            "computation_needed": comp,
            "confidence": conf,
            "action": action,
            "search_filter": dtype if dtype != "general" else None,
            "reason": reason,
        }
