# Auto-generated evaluation dataset for bank regulation PDF
# Generated: 2026-07-20
# Type distribution: 2 clear, 2 vague, 2 typo, 2 synonym, 2 cross_para

EVALUATION_DATASET = [
    {
        "query": "招商银行信息披露管理制度中，定期报告包括哪些类型？",
        "type": "clear",
        "keywords": ["定期报告", "类型", "年度报告", "中期报告", "季度报告"],
        "relevant_doc_indices": [1, 2],
    },
    {
        "query": "根据制度，哪些情形下中期报告的财务报告需要经过审计？",
        "type": "clear",
        "keywords": ["中期报告", "财务报告", "审计", "情形"],
        "relevant_doc_indices": [1],
    },
    {
        "query": "如果公司业绩泄露了，该怎么处理啊？",
        "type": "vague",
        "keywords": ["业绩泄露", "处理", "披露"],
        "relevant_doc_indices": [],
    },
    {
        "query": "那个信息披露的原则是什么，能不能简单说说？",
        "type": "vague",
        "keywords": ["信息披露", "原则"],
        "relevant_doc_indices": [3],
    },
    {
        "query": "临时报告包括哪些试项？",
        "type": "typo",
        "keywords": ["临时报告", "事项"],
        "relevant_doc_indices": [2, 3],
    },
    {
        "query": "董事和高管对定期报告签署意见有什么要求？",
        "type": "typo",
        "keywords": ["董事", "高管", "定期报告", "签署意见"],
        "relevant_doc_indices": [1, 3, 4, 5],
    },
    {
        "query": "按照法规，哪些信息算作重大事件需要发布临时公告？",
        "type": "synonym",
        "keywords": ["重大事件", "临时公告", "临时报告"],
        "relevant_doc_indices": [3],
    },
    {
        "query": "公司进行资产买卖时，披露的标准是什么？",
        "type": "synonym",
        "keywords": ["资产买卖", "交易", "披露标准"],
        "relevant_doc_indices": [1, 2, 3, 5],
    },
    {
        "query": "年度报告和中期报告在披露内容上有什么主要区别？",
        "type": "cross_para",
        "keywords": ["年度报告", "中期报告", "披露内容", "区别"],
        "relevant_doc_indices": [],
    },
    {
        "query": "如果董事对定期报告内容有异议，制度中是如何规定的？同时，定期报告披露前财务数据泄露了怎么办？",
        "type": "cross_para",
        "keywords": ["董事", "异议", "定期报告", "财务数据泄露", "披露"],
        "relevant_doc_indices": [1, 2],
    },
]
