可以。你这个 8 指标版本已经比 6 指标更像一个真正可用的 **Sector Radar / 主线雷达** 了。

我建议总分 **100 分**，分成四大块：

```text
趋势强度：30分
扩散广度：25分
资金状态：15分
核心股结构：30分
```

具体分配如下。

---

# 一、推荐权重版本

| 指标                  | 含义     |      权重 |
| ------------------- | ------ | ------: |
| RS_20               | 短期相对强度 |  **18** |
| RS_60               | 中期相对强度 |  **12** |
| NH_60               | 强扩散    |  **15** |
| Above_MA20          | 趋势扩散   |  **10** |
| Volume_Ratio        | 资金活跃度  |  **10** |
| Leader_Status       | 龙头状态   |  **15** |
| Anchor_Status       | 中军状态   |  **10** |
| Concentration_Ratio | 龙头集中度  |  **10** |
| **合计**              |        | **100** |

这是我最推荐你的 MVP 版本。

---

# 二、为什么这样分？

## 1. RS_20 给 18 分

你的系统平均持股 **7.5 天**，所以短期相对强度非常重要。

```text
RS_20 = 当前市场资金是否正在选择这个方向
```

如果一个方向过去 20 日已经明显弱于市场，即使它是长期大方向，也不适合你当前优先交易。

所以 RS_20 权重要高。

---

## 2. RS_60 给 12 分

RS_60 代表中期趋势背景。

```text
RS_60 = 这个方向是不是中期强势方向
```

它比 RS_20 慢，但可以防止你追一日游热点。

如果：

```text
RS_20 强
RS_60 弱
```

可能只是短期反弹。

如果：

```text
RS_20 强
RS_60 强
```

更像真正主线。

---

## 3. NH_60 给 15 分

NH_60 是**强扩散**，非常关键。

```text
NH_60 = 板块内有多少股票正在创60日新高
```

它说明资金不是只买一只龙头，而是在整个方向扩散。

主线必须有扩散。

所以 NH_60 应该高权重。

---

## 4. Above_MA20 给 10 分

Above_MA20 是**趋势扩散**。

```text
Above_MA20 = 板块内有多少股票还在短期趋势里
```

它没有 NH_60 那么强，但能告诉你板块健康度。

比如：

```text
NH_60 高，Above_MA20 高 = 强主线
NH_60 高，Above_MA20 低 = 龙头独舞
NH_60 低，Above_MA20 高 = 修复/蓄势
NH_60 低，Above_MA20 低 = 弱
```

---

## 5. Volume_Ratio 给 10 分

成交额是资金是否参与的证据。

但它不能单独太高权重，因为：

```text
放量上涨 = 好
放量滞涨 = 风险
放量下跌 = 坏
```

所以 Volume_Ratio 给 **10 分**就够，必须配合价格和扩散一起看。

---

## 6. Leader_Status 给 15 分

主线必须有龙头。

```text
Leader_Status = 龙头是否还在打高度
```

如果龙头不创新高，主线很难维持。

所以它和 NH_60 一样重要。

---

## 7. Anchor_Status 给 10 分

中军代表资金容量和机构稳定性。

```text
Anchor_Status = 大资金是否还愿意稳住这个方向
```

如果小票乱飞，但中军不动，容易是假热闹。

Anchor_Status 给 10 分比较合适。

---

## 8. Concentration_Ratio 给 10 分

这个指标比较特殊。

它不是越高越好，也不是越低越好。

它判断的是：

```text
龙头集中度是否健康
```

你真正想识别的是：

```text
龙头集中 + 扩散增强 = 健康主线
龙头集中 + 扩散下降 = 龙头独舞
集中度下降 + 扩散增强 = 扩散主线
集中度高 + 龙头滞涨 = 拥挤风险
```

所以它适合给 10 分，用作结构修正。

---

# 三、每个指标怎么打分？

下面给你一个可编码版本。

---

## 1. RS_20：18 分

用板块 20 日涨幅相对基准排名。

```text
RS_20 = 板块20日涨幅 - 基准20日涨幅
```

按全市场行业/主题排名打分：

| RS_20 排名 |  得分 |
| -------- | --: |
| 前 10%    |  18 |
| 前 20%    |  15 |
| 前 30%    |  12 |
| 前 40%    |   9 |
| 前 50%    |   6 |
| 后 50%    | 0–5 |

更平滑的写法：

```text
RS_20_score = 18 × percentile_rank(RS_20)
```

这里 percentile_rank 越强越接近 1。

---

## 2. RS_60：12 分

同理：

| RS_60 排名 |  得分 |
| -------- | --: |
| 前 10%    |  12 |
| 前 20%    |  10 |
| 前 30%    |   8 |
| 前 40%    |   6 |
| 前 50%    |   4 |
| 后 50%    | 0–3 |

公式：

```text
RS_60_score = 12 × percentile_rank(RS_60)
```

---

## 3. NH_60：15 分

```text
NH_60 = 创60日新高股票数 / 板块股票总数
```

| NH_60   |  得分 |
| ------- | --: |
| ≥ 25%   |  15 |
| 20%–25% |  13 |
| 15%–20% |  11 |
| 10%–15% |   8 |
| 5%–10%  |   5 |
| < 5%    | 0–3 |

如果你想更平滑：

```text
NH_60_score = min(NH_60 / 25%, 1) × 15
```

---

## 4. Above_MA20：10 分

```text
Above_MA20 = 站上20日线股票数 / 板块股票总数
```

| Above_MA20 | 得分 |
| ---------- | -: |
| ≥ 80%      | 10 |
| 70%–80%    |  8 |
| 60%–70%    |  6 |
| 50%–60%    |  4 |
| 40%–50%    |  2 |
| < 40%      |  0 |

公式：

```text
Above_MA20_score = min(Above_MA20 / 80%, 1) × 10
```

---

## 5. Volume_Ratio：10 分

```text
Volume_Ratio = 今日板块成交额 / 过去20日平均成交额
```

但成交额不能只看放大，要结合板块涨跌。

### 简单版

| Volume_Ratio |       得分 |
| ------------ | -------: |
| 1.5–2.5      |       10 |
| 1.2–1.5      |        8 |
| 1.0–1.2      |        6 |
| 0.8–1.0      |        4 |
| < 0.8        |        2 |
| > 2.5        | 6，需要警惕过热 |

### 更好版本：结合价格

```text
if Volume_Ratio > 1.2 and sector_return > benchmark_return:
    高分
elif Volume_Ratio > 1.5 and sector_return <= benchmark_return:
    降分，可能放量滞涨
elif Volume_Ratio < 0.8:
    低分，资金不活跃
```

建议：

```text
Volume_score = 10分，但放量滞涨最多给5分
```

---

## 6. Leader_Status：15 分

先定义龙头：

```text
龙头 = 板块内60日涨幅Top 3
且成交额排名前30%
```

然后打分：

| 龙头状态                   | 得分 |
| ---------------------- | -: |
| 龙头创 60日/120日新高         | 15 |
| 龙头距离20日高点 < 3%         | 13 |
| 龙头距离20日高点 3%–6%        | 10 |
| 龙头站上 MA20，但距离高点 6%–10% |  7 |
| 龙头跌破 MA20              |  3 |
| 龙头跌破 MA50              |  0 |

如果有多个龙头，取 Top3 平均或最高两个平均：

```text
Leader_Status_score = Top3 龙头状态得分平均
```

我更建议用：

```text
Top2 龙头平均分
```

因为 A 股主线通常有 1–2 个核心龙头，不一定需要很多。

---

## 7. Anchor_Status：10 分

先定义中军：

```text
中军 = 板块内成交额Top 3–5
且市值Top 10
```

然后打分：

| 中军状态                                   | 得分 |
| -------------------------------------- | -: |
| 中军在 MA20/MA50 之上，RS_20 > 0，距20日高点 < 5% | 10 |
| 中军在 MA20/MA50 之上，RS_20 > 0             |  8 |
| 中军在 MA20 之上，但弱于龙头                      |  6 |
| 中军跌破 MA20，但仍在 MA50 上                   |  3 |
| 中军跌破 MA50                              |  0 |

公式化一点：

```text
Anchor_score =
+2 中军 > MA20
+2 中军 > MA50
+2 中军RS_20 > 0
+2 中军距20日高点 < 8%
+2 中军成交额 > 20日均额
```

满分 10。

---

## 8. Concentration_Ratio：10 分

```text
Concentration_Ratio = 板块成交额Top3股票成交额 / 板块总成交额
```

这个指标要和 NH_60、Above_MA20 一起看。

### 健康状态打分

| 状态                               |  得分 |
| -------------------------------- | --: |
| 集中度中高 + NH_60 上升 + Above_MA20 上升 |  10 |
| 集中度适中 + 扩散良好                     |   8 |
| 集中度很高 + 只有龙头强，NH_60 下降           |   4 |
| 集中度很低 + 没有龙头                     |   3 |
| 集中度很高 + 龙头滞涨/放量长阴                | 0–2 |

简单阈值可以这样：

```text
Concentration_Ratio 15%–35% 比较健康
> 40% 说明过度集中，要看是否扩散
< 10% 说明没有核心，主线辨识度不足
```

但不同板块股票数量差异很大，所以最好使用动态分位。

---

# 四、最终主线分数公式

```text
Sector_Leadership_Score =
18 × RS_20_score_norm
+ 12 × RS_60_score_norm
+ 15 × NH_60_score_norm
+ 10 × Above_MA20_score_norm
+ 10 × Volume_score_norm
+ 15 × Leader_Status_score_norm
+ 10 × Anchor_Status_score_norm
+ 10 × Concentration_score_norm
```

其中每个 `score_norm` 都是 0–1。

---

# 五、分类标准

|     总分 | 状态         |
| -----: | ---------- |
| 85–100 | 强主线        |
|  75–85 | 主线         |
|  65–75 | 主线候选 / 强题材 |
|  55–65 | 活跃题材       |
|  45–55 | 热点/反弹      |
|    <45 | 普通/休眠      |

对你的系统，建议交易优先级：

```text
强主线：优先做
主线：正常做
主线候选：观察 + 只做强信号
活跃题材：降低仓位，只做最强票
热点/反弹：不追
休眠：忽略
```

---

# 六、给你一个更贴近 7.5 天持股的版本

因为你的持仓周期短，RS_20、Leader_Status、NH_60 应该更重要。

所以我建议最终采用这个权重：

```text
RS_20：18
RS_60：12
NH_60：15
Above_MA20：10
Volume_Ratio：10
Leader_Status：15
Anchor_Status：10
Concentration_Ratio：10
```

如果你以后发现系统持股周期拉长，比如 20–30 天，可以改成：

```text
RS_20：15
RS_60：15
NH_60：15
Above_MA20：10
Volume_Ratio：10
Leader_Status：15
Anchor_Status：10
Concentration_Ratio：10
```

也就是提高 RS_60，降低 RS_20。

---

# 七、最关键的解释

这 8 个指标其实对应 4 个问题：

```text
1. 方向强不强？
RS_20 + RS_60

2. 是不是扩散？
NH_60 + Above_MA20

3. 有没有资金？
Volume_Ratio + Concentration_Ratio

4. 核心股有没有稳住？
Leader_Status + Anchor_Status
```

一句话：

> **RS 判断方向，NH/MA20 判断扩散，Volume 判断资金，Leader/Anchor 判断主线骨架。**

如果一个方向总分很高，说明：

```text
它不仅涨，
而且强于市场；
不仅龙头强，
而且有扩散；
不仅有热度，
而且中军稳住；
不仅成交活跃，
而且资金结构健康。
```

这才是真正适合你系统优先交易的主线。


可以。这里要把两个概念拆清楚：

> **中军状态 Anchor_Status：看“大资金代表股是否稳住”。**
> **龙头集中度 Concentration_Ratio：看“资金是不是过度集中在少数龙头上”。**

它们解决的问题不一样。

```text
Leader_Status：龙头有没有打高度
Anchor_Status：中军有没有稳住板块
Concentration_Ratio：资金结构是否健康，还是龙头独舞/过度拥挤
```

---

# 一、如何量化中军状态 Anchor_Status

## 1. 先定义什么是中军

在一个申万三级行业里，中军通常不是涨幅最大的股票，而是：

```text
市值大
成交额大
机构能买
走势稳定
能代表板块方向
```

所以中军候选可以这样选：

```text
中军候选 = 行业内
市值排名前 30%
且 最近20日平均成交额排名前 30%
```

更简单的 MVP：

```text
中军候选 = 行业内“市值 Top 5”和“20日平均成交额 Top 5”的交集
```

如果交集太少，可以用并集后打分。

---

# 二、中军状态打分：10 分制

建议 Anchor_Status 总分 **10 分**。

## 指标 1：趋势位置，4 分

看中军是否还在趋势结构里。

```text
中军 > MA20：+2
中军 > MA50：+2
```

解释：

```text
站上 MA20 = 短期趋势未坏
站上 MA50 = 中期趋势未坏
```

如果中军跌破 MA20，是短期警告；跌破 MA50，通常说明主线已经明显降级。

---

## 指标 2：相对强度，2 分

```text
中军20日涨幅 > 基准20日涨幅：+2
```

基准可以用：

```text
沪深300 / 中证全指 / 万得全A / 所属一级行业指数
```

更推荐用 **万得全A或中证全指**，因为你要判断它是否强于全市场。

---

## 指标 3：距离高点，2 分

```text
中军距离20日高点回撤 < 5%：+2
中军距离20日高点回撤 5%–10%：+1
中军距离20日高点回撤 > 10%：+0
```

公式：

```text
Drawdown_20 = close / rolling_high_20 - 1
```

比如：

```text
close = 95
20日最高 = 100
Drawdown_20 = -5%
```

中军距离高点越近，说明资金还在维护。

---

## 指标 4：成交额状态，2 分

```text
今日成交额 > 20日平均成交额：+1
5日平均成交额 > 20日平均成交额：+1
```

解释：

```text
今日放量 = 当日有资金关注
5日放量 = 资金持续关注
```

如果中军缩量下跌、反弹无量，说明大资金参与度下降。

---

# 三、Anchor_Status 公式

单只中军得分：

```text
Anchor_Score_Stock =
2 × I(close > MA20)
+ 2 × I(close > MA50)
+ 2 × I(RS_20 > 0)
+ Drawdown_Score
+ Volume_Score
```

其中：

```text
Drawdown_Score:
0% ~ -5%：2分
-5% ~ -10%：1分
< -10%：0分

Volume_Score:
今日成交额 > 20日均额：+1
5日均成交额 > 20日均额：+1
```

总分 10 分。

行业 Anchor_Status 可以取中军候选的平均值：

```text
Anchor_Status = mean(Top 3 Anchor_Score_Stock)
```

或者更稳一点，用成交额加权：

```text
Anchor_Status = weighted_average(Anchor_Score_Stock, weight = amount_20d_avg)
```

---

# 四、Anchor_Status 如何解释

| Anchor_Status | 状态   | 含义           |
| ------------: | ---- | ------------ |
|          8–10 | 中军强  | 大资金仍在维护，主线健康 |
|           6–8 | 中军正常 | 板块还能看，但要观察   |
|           4–6 | 中军走弱 | 主线开始分化       |
|           2–4 | 中军失守 | 主线降级风险大      |
|           0–2 | 中军破位 | 退出优先观察池      |

实战上：

```text
龙头强 + 中军强 = 真主线
龙头强 + 中军弱 = 龙头独舞，谨慎
龙头弱 + 中军强 = 板块防守/蓄势
龙头弱 + 中军弱 = 退潮
```

---

# 五、如何量化龙头集中度 Concentration_Ratio

## 1. 最基础公式

```text
Concentration_Ratio = 行业内成交额Top3股票成交额之和 / 行业总成交额
```

也可以用 Top5：

```text
Top5_Concentration = 行业内成交额Top5股票成交额之和 / 行业总成交额
```

对于申万三级行业，通常股票数量不多，所以我建议用：

```text
Top3_Concentration
```

如果行业内股票很多，可以用 Top5。

---

# 六、但集中度不是越高越好

这是最容易误解的地方。

集中度高有两种含义：

## 健康集中

```text
龙头强
中军强
扩散也在增强
```

说明资金先买核心，再向板块扩散。

## 危险集中

```text
只有龙头强
NH_60 下降
Above_MA20 下降
后排掉队
```

说明龙头独舞，板块已经不扩散。

所以 Concentration_Ratio 必须和：

```text
NH_60
Above_MA20
Leader_Status
Anchor_Status
```

一起看。

---

# 七、Concentration_Ratio 的四种状态

假设：

```text
Concentration_Ratio = Top3成交额 / 行业总成交额
```

## 1. 健康主线

```text
Concentration_Ratio 中等偏高
NH_60 上升
Above_MA20 上升
Leader_Status 强
Anchor_Status 强
```

含义：

> 核心股带队，板块正在扩散。

这是最理想状态。

---

## 2. 龙头独舞

```text
Concentration_Ratio 高
Leader_Status 强
但 NH_60 下降
Above_MA20 下降
Anchor_Status 弱
```

含义：

> 只剩少数龙头硬撑，后排不跟。

这是主线分化信号。

---

## 3. 群龙无首

```text
Concentration_Ratio 很低
NH_60 不高
Above_MA20 一般
Leader_Status 弱
```

含义：

> 板块没有核心，资金没有形成共识。

这通常不是主线。

---

## 4. 过度拥挤

```text
Concentration_Ratio 极高
Volume_Ratio 高
Leader_Status 开始下降
龙头放量滞涨/长上影
```

含义：

> 资金挤在少数股票里，但价格开始不跟了。

这是高位风险信号。

---

# 八、Concentration_Ratio 如何打分：10 分制

因为它不是越高越好，所以不要简单线性打分。

建议用“结构打分”。

## 版本 A：简单阈值打分

| 状态                |  得分 |
| ----------------- | --: |
| 集中度适中，且扩散增强       |  10 |
| 集中度偏高，但龙头/中军/扩散都强 |   8 |
| 集中度偏低，但广度不错       |   6 |
| 集中度很高，扩散下降        |   4 |
| 集中度很低，没有龙头        |   3 |
| 集中度极高，龙头滞涨        | 0–2 |

可以这样编码：

```text
if 0.15 <= CR <= 0.35 and NH_60_trend > 0 and Above_MA20_trend > 0:
    score = 10
elif CR > 0.35 and Leader_Status >= 12 and Anchor_Status >= 7 and NH_60_trend >= 0:
    score = 8
elif CR < 0.15 and Above_MA20 > 0.6:
    score = 6
elif CR > 0.40 and NH_60_trend < 0:
    score = 4
elif CR < 0.10 and Leader_Status < 8:
    score = 3
elif CR > 0.45 and Leader_Status declining:
    score = 0~2
```

---

## 版本 B：更推荐，用历史分位

不同申万三级行业股票数量差异很大，所以固定阈值可能不准。

更好的方式：

```text
CR_Pctl_120 = 当前 Concentration_Ratio 在过去120日的历史分位
```

然后结合扩散判断。

| CR历史分位  | 扩散状态        | 解释        |
| ------- | ----------- | --------- |
| 60%–85% | NH/MA20上升   | 健康集中      |
| >90%    | NH/MA20下降   | 过度集中      |
| <30%    | Leader弱     | 无核心       |
| <30%    | Above_MA20高 | 广泛修复，但缺龙头 |

---

# 九、我推荐你的最终 Concentration_Score

用下面这个组合最实用：

```text
Concentration_Score = 10分
```

规则：

```text
基础分 = 5
```

加分：

```text
CR在过去120日的40%–85%分位：+2
NH_60最近5日上升：+1
Above_MA20最近5日上升：+1
Leader_Status >= 12：+1
```

扣分：

```text
CR > 90%分位 且 NH_60下降：-2
CR > 90%分位 且 Leader_Status下降：-2
CR < 20%分位 且 Leader_Status < 8：-2
Anchor_Status < 5：-1
```

最后限制在 0–10：

```text
Concentration_Score = max(0, min(10, score))
```

这样更合理，因为它判断的是“资金结构健康度”，不是纯集中度大小。

---

# 十、完整 Python 伪代码

## Anchor_Status

```python
def stock_anchor_score(close, ma20, ma50, ret20, bench_ret20,
                       high20, amount_today, amount_ma5, amount_ma20):
    score = 0

    if close > ma20:
        score += 2
    if close > ma50:
        score += 2
    if ret20 > bench_ret20:
        score += 2

    drawdown20 = close / high20 - 1
    if drawdown20 >= -0.05:
        score += 2
    elif drawdown20 >= -0.10:
        score += 1

    if amount_today > amount_ma20:
        score += 1
    if amount_ma5 > amount_ma20:
        score += 1

    return max(0, min(10, score))
```

行业中军得分：

```python
def sector_anchor_status(anchor_stocks):
    # anchor_stocks: 中军候选股票列表，每只股票有 anchor_score 和 amount_ma20
    total_amount = sum(s.amount_ma20 for s in anchor_stocks)
    if total_amount == 0:
        return 0

    return sum(s.anchor_score * s.amount_ma20 for s in anchor_stocks) / total_amount
```

---

## Concentration_Ratio

```python
def concentration_ratio(sector_stocks):
    total_amount = sum(s.amount_today for s in sector_stocks)
    if total_amount == 0:
        return 0

    top3 = sorted(sector_stocks, key=lambda s: s.amount_today, reverse=True)[:3]
    top3_amount = sum(s.amount_today for s in top3)

    return top3_amount / total_amount
```

---

## Concentration_Score

```python
def concentration_score(cr, cr_pctl_120, nh60_trend, above_ma20_trend,
                        leader_status, leader_status_trend, anchor_status):
    score = 5

    # 健康集中区
    if 0.40 <= cr_pctl_120 <= 0.85:
        score += 2

    # 扩散增强
    if nh60_trend > 0:
        score += 1
    if above_ma20_trend > 0:
        score += 1

    # 龙头强
    if leader_status >= 12:
        score += 1

    # 过度集中 + 扩散下降
    if cr_pctl_120 > 0.90 and nh60_trend < 0:
        score -= 2

    # 过度集中 + 龙头走弱
    if cr_pctl_120 > 0.90 and leader_status_trend < 0:
        score -= 2

    # 无核心
    if cr_pctl_120 < 0.20 and leader_status < 8:
        score -= 2

    # 中军弱
    if anchor_status < 5:
        score -= 1

    return max(0, min(10, score))
```

---

# 十一、最终使用方式

在你的 100 分主线雷达里：

```text
Anchor_Status：10分
Concentration_Ratio：10分
```

它们的意义是：

```text
Anchor_Status 高：
说明中军稳，板块骨架没坏。

Concentration_Score 高：
说明资金结构健康，既有核心，又没有过度拥挤。

Anchor_Status 低：
说明中军失守，主线容易降级。

Concentration_Score 低：
说明要么龙头独舞，要么没有龙头，要么过度拥挤。
```

---

# 十二、最实用的判断组合

## 真主线

```text
Leader_Status 高
Anchor_Status 高
NH_60 上升
Concentration_Score 高
```

## 龙头独舞

```text
Leader_Status 高
Anchor_Status 低
NH_60 下降
Concentration_Ratio 极高
```

## 主线扩散

```text
Leader_Status 高
Anchor_Status 高
NH_60 上升
Above_MA20 上升
Concentration_Ratio 适中
```

## 主线退潮

```text
Leader_Status 下降
Anchor_Status 下降
NH_60 下降
Above_MA20 下降
Concentration_Ratio 高位但价格滞涨
```

一句话：

> **中军状态看板块骨架是否稳定；龙头集中度看资金结构是否健康。中军稳 + 龙头强 + 扩散升，才是真正强主线。**
