from typing import List, Dict, Set

EVALUATION_DATASET: List[dict] = [
    # === 运营战略 ===
    {
        "query": "什么是企业使命",
        "relevant_contents": ["使命", "企业使命"],
        "doc_type": "concept",
    },
    {
        "query": "战略管理的过程包括哪些步骤",
        "relevant_contents": ["战略管理过程", "战略分析", "战略选择", "战略实施"],
        "doc_type": "procedure",
    },
    {
        "query": "明茨伯格对战略的定义是什么",
        "relevant_contents": ["明茨伯格", "计划", "计谋", "模式", "定位", "观念"],
        "doc_type": "concept",
    },
    {
        "query": "什么是三角底线模型",
        "relevant_contents": ["三角底线", "经济责任", "社会责任", "环境责任"],
        "doc_type": "concept",
    },
    {
        "query": "特斯拉的企业使命是什么",
        "relevant_contents": ["特斯拉", "加速世界向可持续交通的转型"],
        "doc_type": "faq",
    },
    {
        "query": "腾讯的使命经历了哪些变化",
        "relevant_contents": ["腾讯", "用户依赖的朋友", "通过互联网服务提升人类生活品质", "用户为本，科技向善"],
        "doc_type": "faq",
    },

    # === 库存管理 ===
    {
        "query": "库存可以分为哪些类型",
        "relevant_contents": ["周转库存", "安全库存", "调节库存", "在途库存", "投机库存"],
        "doc_type": "concept",
    },
    {
        "query": "库存会带来哪些问题",
        "relevant_contents": ["占用资金", "持有成本", "过期风险", "破损"],
        "doc_type": "concept",
    },
    {
        "query": "库存控制的目标是什么",
        "relevant_contents": ["库存控制的目标", "库存成本", "顾客服务水平"],
        "doc_type": "concept",
    },
    {
        "query": "什么是单周期库存模型",
        "relevant_contents": ["单周期库存模型", "一次性的", "不重复订货"],
        "doc_type": "concept",
    },
    {
        "query": "什么是多周期库存模型",
        "relevant_contents": ["多周期库存模型", "重复订货", "不断地补充"],
        "doc_type": "concept",
    },
    {
        "query": "边际分析法如何确定最佳订货量",
        "relevant_contents": ["边际分析法", "最佳订货量", "期望收益"],
        "doc_type": "calculation",
    },
    {
        "query": "什么是安全库存",
        "relevant_contents": ["安全库存", "不确定因素"],
        "doc_type": "concept",
    },
    {
        "query": "库存ABC管理是什么",
        "relevant_contents": ["库存ABC管理"],
        "doc_type": "concept",
    },

    # === 质量管理 ===
    {
        "query": "设计质量和符合性质量有什么区别",
        "relevant_contents": ["设计质量", "符合性质量", "一致性质量"],
        "doc_type": "concept",
    },
    {
        "query": "设计质量的维度有哪些",
        "relevant_contents": ["设计质量的维度", "性能", "特征", "可靠性", "耐久性", "维护性", "美观", "感知质量"],
        "doc_type": "concept",
    },
    {
        "query": "什么是质量成本",
        "relevant_contents": ["质量成本", "零缺陷"],
        "doc_type": "concept",
    },
    {
        "query": "因果分析图如何使用",
        "relevant_contents": ["因果分析图", "复印不清楚"],
        "doc_type": "method",
    },
    {
        "query": "什么是散点图",
        "relevant_contents": ["散点图", "散布图", "相关图"],
        "doc_type": "concept",
    },
    {
        "query": "帕累托图如何绘制",
        "relevant_contents": ["帕累托图", "频数", "累积频率"],
        "doc_type": "method",
    },
    {
        "query": "质量管理的主要工具有哪些",
        "relevant_contents": ["质量管理工具", "统计过程控制", "过程能力分析", "6σ管理"],
        "doc_type": "faq",
    },
    {
        "query": "什么是6σ管理",
        "relevant_contents": ["6σ管理"],
        "doc_type": "concept",
    },

    # === 需求预测 ===
    {
        "query": "定性预测和定量预测有什么区别",
        "relevant_contents": ["定性预测方法", "定量预测方法", "数学方法"],
        "doc_type": "concept",
    },
    {
        "query": "时间序列模型和因果关系模型有什么不同",
        "relevant_contents": ["时间序列模型", "因果关系模型", "变量之间"],
        "doc_type": "concept",
    },
    {
        "query": "需求预测的一般步骤是什么",
        "relevant_contents": ["需求预测的一般步骤", "明确预测的目的", "确定时间跨度", "选择适当的预测方法"],
        "doc_type": "procedure",
    },
    {
        "query": "加权移动平均法有什么特点",
        "relevant_contents": ["加权移动平均法", "响应性", "权重"],
        "doc_type": "method",
    },
    {
        "query": "简单移动平均法和加权移动平均法有什么区别",
        "relevant_contents": ["加权移动平均法", "简单移动平均法", "权重"],
        "doc_type": "concept",
    },
    {
        "query": "如何选择预测模型",
        "relevant_contents": ["预测模型选择", "预测的时间跨度", "响应性", "抗干扰性"],
        "doc_type": "method",
    },

    # === 设施选址 ===
    {
        "query": "设施选址有哪些类型",
        "relevant_contents": ["选址问题", "生产设施选址", "服务设施选址", "单一设施选址", "多设施选址", "连续选址", "离散选址"],
        "doc_type": "concept",
    },
    {
        "query": "设施选址的步骤是什么",
        "relevant_contents": ["选址步骤", "明确选址的目标", "分析影响因素", "评估并作出选择"],
        "doc_type": "procedure",
    },
    {
        "query": "工艺原则布置和产品原则布置有什么区别",
        "relevant_contents": ["工艺原则布置", "产品原则布置"],
        "doc_type": "concept",
    },
    {
        "query": "如何进行多仓库选址的运输成本优化",
        "relevant_contents": ["运输成本", "工厂到各仓库的单位运费", "新仓库"],
        "doc_type": "calculation",
    },

    # === 运营战略 ===
    {
        "query": "运营战略在企业战略体系中处于什么位置",
        "relevant_contents": ["运营战略", "企业战略体系"],
        "doc_type": "concept",
    },
    {
        "query": "战略环境分析包括哪些内容",
        "relevant_contents": ["内部条件", "外部环境", "战略选择"],
        "doc_type": "procedure",
    },

    # === 质量管理 ===
    {
        "query": "统计过程控制的作用是什么",
        "relevant_contents": ["统计过程控制"],
        "doc_type": "concept",
    },
]
