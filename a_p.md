一、先定义 Winner / Loser

先别用“盈利/亏损”这么粗，因为 0R、微利、微亏都很吵。

我建议你先分 4 档：

Strong Winner: pnl_r >= 2
Winner:        0 < pnl_r < 2
Loser:        -1 < pnl_r <= 0
Strong Loser:  pnl_r <= -1

真正重点分析的是：

Strong Winner vs Strong Loser

因为这两组最能反映“真正的 edge”。

如果你样本不够，也可以先简化成：

Winner: pnl_r > 0
Loser:  pnl_r <= 0

但最优先还是强赢家 vs 强输家。

二、你要准备哪些字段

每一笔交易尽量导出这些列。你已经有不少了。

1. Detect / Day1 强度
gap_pct
day1_change_pct
day1_close_strength
day1_volume_ratio
day1_range_pct
2. Day2 回踩结构
pullback_pct
price_up_ratio
entry_time
entry_vs_day2_low
entry_vs_day2_high
3. Day2 午后行为
has_long_lower_shadow
shadow_ratio
stabilized_after_1400
no_new_low_after_1400
close_vs_1400_open
close_vs_low_after_1400
4. 量价承接
vol_ratio_14_30
price_vs_vwap
close_vs_vwap
vwap_slope_pm
afternoon_volume_trend
5. 结果标签
pnl_r
max_favorable_excursion_r
max_adverse_excursion_r
exit_reason
hold_days
三、自动分析的核心逻辑

我们要找的不是“均值有点不同”，而是：

Winner 和 Loser 之间，哪些特征差异大、方向稳定、可转成规则。

所以每个特征都做三件事：

1. 对比均值/中位数

比如：

Winner 的 pullback_pct 均值
Loser 的 pullback_pct 均值
2. 对比分布区间

比如：

Winner 的 25%、50%、75%
Loser 的 25%、50%、75%
3. 形成可执行阈值

比如：

Winner 的 price_vs_vwap > 0 占比 82%
Loser 只有 31%

那就很可能能变成：

过滤条件：entry时 price >= VWAP
四、最值得优先分析的 8 个特征

我建议你先跑这 8 个，因为最可能有用。

1. price_vs_vwap

最优先。

看：

Winner 中有多少比例 >= 0
Loser 中有多少比例 < 0

如果差异明显，直接上规则。

2. no_new_low_after_1400

很可能是强特征。

看：

Winner 中这个标记为 1 的比例
Loser 中这个标记为 1 的比例
3. pullback_pct

这是核心结构变量。

看 Winner vs Loser 的中位数差异。
如果 Winner 明显更浅，说明应收紧回踩深度。

4. vol_ratio_14_30

你已经知道 volume 很关键，但不是越低越好，要找到“甜区”。

看是否：

Winner 集中在 0.45–0.70
Loser 集中在 0.70–0.95
5. close_vs_vwap

比 price_vs_vwap 更稳定一点，适合做确认。

6. close_vs_low_after_1400

能衡量午后承接强度。

7. gap_pct

看看是不是 gap 太小的交易更差。

8. day1_close_strength

看看 detect day 本身够不够强。

五、自动分析输出应该长什么样

你最终想得到这种表。

feature	Winner median	Loser median	diff	推荐规则
price_vs_vwap	0.8%	-1.2%	大	entry 时 price >= VWAP
pullback_pct	0.34	0.58	大	pullback <= 0.45
vol_ratio_14_30	0.61	0.79	中	volume ratio <= 0.72
no_new_low_after_1400	87%	42%	大	必须 no new low
gap_pct	4.8%	2.6%	中	gap >= 3%

这张表出来后，你就能知道：

哪些条件应该加强
哪些条件应该删除
哪些条件应该改成区间
六、推荐过滤条件的生成逻辑

别直接用最优值，容易过拟合。
应该用这种方法：

连续变量

例如 pullback_pct

取 Winner 组的 75 分位、Loser 组的中位数
在两者之间选一个稳一点的值

比如：

Winner Q75 = 0.46
Loser median = 0.57

那推荐先试：

pullback_pct <= 0.50
二值变量

例如 no_new_low_after_1400

Winner 87%
Loser 42%

那推荐直接试：

必须 no_new_low_after_1400 == 1
区间变量

例如 vol_ratio_14_30

如果发现：

太高不好
太低也一般

那就别写成单边规则，而是：

0.45 <= vol_ratio_14_30 <= 0.72