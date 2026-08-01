-- ============================================================================
-- Bank RAG — Sample Financial Product Schema
-- ============================================================================
-- Populates the bank_rag database with structured tables for Text-to-SQL.
-- Run: mysql -u root -pABC123 bank_rag < bank_schema.sql
-- ============================================================================

-- 1. Deposit products (存款产品)
CREATE TABLE IF NOT EXISTS deposit_products (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    product_name    VARCHAR(100)    NOT NULL COMMENT '产品名称',
    product_type    VARCHAR(30)     NOT NULL COMMENT '类型: 活期/定期/大额存单/通知存款',
    term_months     INT             COMMENT '期限(月), 活期为NULL',
    annual_rate     DECIMAL(6,4)    NOT NULL COMMENT '年化利率(小数, 0.0175=1.75%)',
    min_deposit     DECIMAL(14,2)   COMMENT '最低起存金额',
    early_withdraw  VARCHAR(50)     COMMENT '提前支取规则',
    is_active       TINYINT(1)      DEFAULT 1 COMMENT '是否在售',
    updated_at      DATE            COMMENT '利率生效日期',
    notes           VARCHAR(200)    COMMENT '备注'
) COMMENT '存款产品利率表';

-- 2. Loan products (贷款产品)
CREATE TABLE IF NOT EXISTS loan_products (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    product_name    VARCHAR(100)    NOT NULL COMMENT '产品名称',
    loan_type       VARCHAR(30)     NOT NULL COMMENT '类型: 房贷/消费贷/经营贷/信用贷',
    rate_type       VARCHAR(20)     NOT NULL COMMENT '利率类型: 固定/浮动(LPR)',
    base_rate       DECIMAL(6,4)    NOT NULL COMMENT '基础年利率',
    min_amount      DECIMAL(14,2)   COMMENT '最低贷款金额',
    max_amount      DECIMAL(14,2)   COMMENT '最高贷款金额',
    max_term_months INT             COMMENT '最长贷款期限(月)',
    prepay_penalty  VARCHAR(100)    COMMENT '提前还款违约金规则',
    requirements    VARCHAR(300)    COMMENT '申请条件',
    is_active       TINYINT(1)      DEFAULT 1,
    updated_at      DATE
) COMMENT '贷款产品表';

-- 3. Service fees (手续费标准)
CREATE TABLE IF NOT EXISTS service_fees (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    service_name    VARCHAR(100)    NOT NULL COMMENT '服务名称',
    fee_type        VARCHAR(30)     NOT NULL COMMENT '收费类型: 固定/按比例/按笔/免费',
    fee_amount      DECIMAL(10,2)   COMMENT '固定金额(元)',
    fee_rate        DECIMAL(6,4)    COMMENT '费率(小数)',
    fee_cap         DECIMAL(10,2)   COMMENT '封顶金额(元)',
    conditions      VARCHAR(200)    COMMENT '收费条件/减免条件',
    applicable_to   VARCHAR(50)     COMMENT '适用对象: 个人/对公/全部',
    updated_at      DATE
) COMMENT '手续费标准表';

-- 4. Bank branches (网点)
CREATE TABLE IF NOT EXISTS branches (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    branch_name     VARCHAR(100)    NOT NULL COMMENT '网点名称',
    district        VARCHAR(50)     NOT NULL COMMENT '所在区域',
    address         VARCHAR(200)    NOT NULL COMMENT '详细地址',
    phone           VARCHAR(20)     COMMENT '联系电话',
    business_hours  VARCHAR(100)    COMMENT '营业时间',
    has_parking     TINYINT(1)      DEFAULT 0 COMMENT '是否有停车位',
    has_wifi        TINYINT(1)      DEFAULT 1 COMMENT '是否有WiFi',
    is_weekend_open TINYINT(1)      DEFAULT 0 COMMENT '周末是否营业'
) COMMENT '网点信息表';

-- ============================================================================
-- Seed data
-- ============================================================================

-- Deposit products
INSERT INTO deposit_products (product_name, product_type, term_months, annual_rate, min_deposit, early_withdraw, updated_at) VALUES
('活期存款',        '活期',     NULL,   0.0020,     1.00,       '随时支取',          '2026-01-01'),
('三个月定期',      '定期',     3,      0.0135,    50.00,       '按活期计息',        '2026-01-01'),
('六个月定期',      '定期',     6,      0.0155,    50.00,       '按活期计息',        '2026-01-01'),
('一年定期',        '定期',     12,     0.0175,    50.00,       '按活期计息',        '2026-01-01'),
('二年定期',        '定期',     24,     0.0225,    50.00,       '按活期计息',        '2026-01-01'),
('三年定期',        '定期',     36,     0.0275,    50.00,       '按活期计息',        '2026-01-01'),
('五年定期',        '定期',     60,     0.0275,    50.00,       '按活期计息',        '2026-01-01'),
('大额存单A(20万)', '大额存单', 12,     0.0210, 200000.00,       '不可提前支取',      '2026-01-01'),
('大额存单B(50万)', '大额存单', 12,     0.0225, 500000.00,       '不可提前支取',      '2026-01-01'),
('大额存单C(100万)','大额存单', 36,     0.0310,1000000.00,       '满1年可按1年期利率计息','2026-01-01'),
('通知存款(一天)',  '通知存款', NULL,    0.0080, 50000.00,       '提前一天通知',      '2026-01-01'),
('通知存款(七天)',  '通知存款', NULL,    0.0135, 50000.00,       '提前七天通知',      '2026-01-01');

-- Loan products
INSERT INTO loan_products (product_name, loan_type, rate_type, base_rate, min_amount, max_amount, max_term_months, prepay_penalty, requirements, updated_at) VALUES
('首套房贷(LPR)',       '房贷',   '浮动(LPR)', 0.0395,  100000.00, 5000000.00, 360, '满1年后免违约金',         '本地户籍或连续缴纳社保满2年', '2026-01-01'),
('二套房贷(LPR)',       '房贷',   '浮动(LPR)', 0.0450,  100000.00, 4000000.00, 300, '满1年后免违约金',         '本地户籍或连续缴纳社保满2年', '2026-01-01'),
('个人消费贷',          '消费贷', '固定',      0.0480,   10000.00,  500000.00,  60, '剩余本金1%违约金',        '信用评分600以上',             '2026-01-01'),
('个人经营贷',          '经营贷', '浮动(LPR)', 0.0385,   50000.00, 3000000.00, 120, '满2年后免违约金',         '营业执照满1年',               '2026-01-01'),
('信用贷(优质客户)',    '信用贷', '固定',      0.0350,   10000.00,  300000.00,  36, '无违约金',               '本行代发工资客户',            '2026-01-01'),
('公积金贷款',          '房贷',   '固定',      0.0310,         NULL,  600000.00, 360, '无违约金',               '连续缴存公积金满6个月',       '2026-01-01');

-- Service fees
INSERT INTO service_fees (service_name, fee_type, fee_amount, fee_rate, fee_cap, conditions, applicable_to, updated_at) VALUES
('同行转账',            '免费',   0.00,       NULL,      NULL,      '柜台/网银/手机银行',            '全部', '2026-01-01'),
('跨行转账(<5万)',      '按笔',   2.00,       NULL,      NULL,      '柜台办理',                      '个人', '2026-01-01'),
('跨行转账(5万-10万)',  '按笔',   5.00,       NULL,      NULL,      '柜台办理',                      '个人', '2026-01-01'),
('跨行转账(>10万)',     '按比例', NULL,       0.0003,    50.00,     '柜台办理, 最高50元封顶',         '个人', '2026-01-01'),
('手机银行跨行转账',    '免费',   0.00,       NULL,      NULL,      '手机银行办理',                  '个人', '2026-01-01'),
('跨境汇款',            '按比例', NULL,       0.0010,    200.00,    '电报费另收80元/笔',            '个人', '2026-01-01'),
('信用卡年费(普卡)',    '固定',   100.00,     NULL,      NULL,      '首年免年费, 刷6次免次年',       '个人', '2026-01-01'),
('信用卡年费(金卡)',    '固定',   300.00,     NULL,      NULL,      '首年免年费, 刷12次免次年',      '个人', '2026-01-01'),
('信用卡取现手续费',    '按比例', NULL,       0.0100,    100.00,    '境内, 最低10元/笔',             '个人', '2026-01-01'),
('短信提醒服务',        '固定',   2.00,       NULL,      NULL,      '每月, 可申请减免',              '个人', '2026-01-01'),
('小额账户管理费',      '固定',   3.00,       NULL,      NULL,      '日均余额<300元收取, 每季度',    '个人', '2026-01-01');

-- Branches
INSERT INTO branches (branch_name, district, address, phone, business_hours, has_parking, has_wifi, is_weekend_open) VALUES
('总行营业部',          '福田区', '深圳市福田区深南大道7088号',   '0755-88888888', '周一至周五 9:00-17:00',         1, 1, 0),
('南山支行',            '南山区', '深圳市南山区科技园南路1号',     '0755-88888801', '周一至周五 9:00-17:00',         1, 1, 0),
('罗湖支行',            '罗湖区', '深圳市罗湖区深南东路5002号',    '0755-88888802', '周一至周五 9:00-17:00',         0, 1, 0),
('宝安支行',            '宝安区', '深圳市宝安区宝安大道168号',     '0755-88888803', '周一至周五 9:00-17:00',         1, 1, 1),
('龙岗支行',            '龙岗区', '深圳市龙岗区龙翔大道2001号',    '0755-88888804', '周一至周五 9:00-17:00',         1, 1, 1),
('福田中心支行',        '福田区', '深圳市福田区华强北路1002号',    '0755-88888805', '周一至周日 10:00-18:00',       0, 1, 1);
