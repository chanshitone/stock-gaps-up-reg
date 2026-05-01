🚀 五、具体实现（你可以直接写代码）
Step 1：定义阈值范围
thresholds = [-0.5, -0.2, 0, 0.2, 0.5, 1.0]  # 单位 %
Step 2：循环回测
results = []

for t in thresholds:
    filtered = trades[trades['index_change'] > t]

    total_r = filtered['pnl_r'].sum()
    trades_n = len(filtered)

    equity = filtered['pnl_r'].cumsum()
    max_dd = (equity.cummax() - equity).max()

    score = total_r / (max_dd + 1e-6)

    results.append((t, total_r, max_dd, trades_n, score))
Step 3：画图
x轴：threshold
y轴：
total_r
max_dd
score