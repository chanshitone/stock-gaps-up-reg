# run_sector_radar.py 流程结构说明

`run_sector_radar.py` 是一个很薄的入口文件：

```python
from src.stock_gaps_reg.sector_radar import main

if __name__ == "__main__":
    main()
```

真正流程都在：

```text
src/stock_gaps_reg/sector_radar.py
```

## 1. 启动方式

示例：

```bash
python run_sector_radar.py --stocks inputs/sector_radar.smoke.csv
```

可选参数：

```bash
python run_sector_radar.py \
  --stocks inputs/sector_radar.smoke.csv \
  --as-of 2026-06-28 \
  --config config/sector_radar.yaml \
  --strategy-config config/strategy.yaml \
  --output-dir outputs/sector_radar
```

参数含义：

| 参数 | 作用 |
|---|---|
| `--stocks` | 输入股票列表，必须包含 `ts_code` 列 |
| `--as-of` | 雷达计算日期，默认今天 |
| `--config` | Sector Radar 配置文件 |
| `--strategy-config` | 用来读取数据缓存目录 |
| `--output-dir` | 输出目录 |

## 2. 主流程

入口调用：

```python
main()
```

主流程顺序如下：

```text
main()
  ↓
解析命令行参数
  ↓
读取 config/sector_radar.yaml
  ↓
读取 config/strategy.yaml，确定 cache_dir
  ↓
初始化 TushareClient
  ↓
读取股票列表
  ↓
构建 sector radar
  ↓
创建本次运行输出目录
  ↓
写出 CSV + Markdown 报告
```

## 3. 配置加载

函数：

```python
load_radar_config(args.config.resolve())
```

读取：

```text
config/sector_radar.yaml
```

当前配置示例：

```yaml
history_calendar_days: 240

short_return_days: 5
medium_return_days: 20

fast_ma_days: 20
slow_ma_days: 60
volatility_days: 20

phase_a_min_short_return_pct: 2.0
phase_a_min_medium_return_pct: 5.0

phase_c_max_medium_return_pct: -3.0
```

这些参数决定：

- 用多少天历史行情
- 短周期收益率窗口
- 中周期收益率窗口
- 快慢均线窗口
- 波动率窗口
- Phase A / Phase C 的阈值

## 4. 股票列表读取

函数：

```python
load_stock_list(args.stocks.resolve())
```

支持两种输入：

### CSV

必须有：

```csv
ts_code
600519.SH
000858.SZ
601318.SH
```

### 文本文件

也可以是用空格、换行、逗号、分号分隔的代码。

读取后会做：

```text
原始代码
  ↓
normalize_ts_code()
  ↓
去空值
  ↓
去重
  ↓
得到 stock_codes
```

## 5. 行业映射逻辑

核心函数：

```python
build_sector_radar(stock_codes, client, args.as_of, radar_config)
```

第一步读取申万一级行业目录：

```python
client.get_sw_index_classify(level="L1", src="SW2021")
```

然后逐个股票查询申万行业归属：

```python
client.get_sw_memberships(ts_code)
```

再根据 `as_of` 日期选择当日有效的行业归属：

```python
select_membership(memberships, as_of)
```

判断规则：

```text
in_date <= as_of
并且
out_date 为空 或 out_date >= as_of
```

如果找不到有效行业，会进入：

```text
unmatched_stocks.csv
```

## 6. 行业行情计算

对每个被映射到的申万一级行业：

```python
client.get_sw_daily(sector_code, start_date, as_of)
```

其中：

```python
start_date = as_of - history_calendar_days
```

然后调用：

```python
calculate_sector_metrics(history, config)
```

计算指标：

| 指标 | 说明 |
|---|---|
| `latest_close` | 最新收盘价 |
| `return_5d_pct` | 短周期收益率，默认 5 日 |
| `return_20d_pct` | 中周期收益率，默认 20 日 |
| `ma20` | 快均线，默认 20 日 |
| `ma60` | 慢均线，默认 60 日 |
| `close_vs_ma20_pct` | 收盘价相对 MA20 的偏离 |
| `ma20_vs_ma60_pct` | MA20 相对 MA60 的偏离 |
| `volatility_20d_ann_pct` | 年化波动率 |
| `score` | 综合评分 |
| `phase` | A/B/C/D 赛道阶段 |

## 7. Phase A/B/C/D 定义

当前代码里的定义是：

### Phase A：强趋势赛道

```text
close > MA20 > MA60
并且
5日收益率 > phase_a_min_short_return_pct
并且
20日收益率 > phase_a_min_medium_return_pct
```

含义：

```text
价格在快均线上方
快均线在慢均线上方
短中期动量都满足阈值
```

### Phase B：分化赛道

```text
close > MA60
并且
close <= MA20 或 5日收益率 <= 0
```

含义：

```text
大结构还没坏
但短期已经开始走弱或震荡
```

### Phase C：失败赛道

```text
MA20 < MA60
并且
20日收益率 < phase_c_max_medium_return_pct
```

含义：

```text
中期均线结构转弱
并且中期收益为负到达阈值
```

### Phase D：无结构赛道

```text
不满足 A/B/C 的其他情况
```

含义：

```text
没有明确趋势结构
不适合强行归类
```

## 8. Score 排名逻辑

当前评分公式：

```python
score =
    0.40 * medium_return
  + 0.25 * short_return
  + 0.20 * above_fast
  + 0.15 * fast_vs_slow
  - 0.05 * volatility
```

也就是：

```text
20日收益率权重最高
其次是5日收益率
再看价格是否强于MA20
再看MA20是否强于MA60
最后扣除波动率
```

排序方式：

```text
Phase A
  ↓
Phase B
  ↓
Phase C
  ↓
Phase D
```

同一个 Phase 内：

```text
score 越高排名越靠前
```

## 9. 输出文件

每次运行会创建一个新的运行目录：

```text
outputs/sector_radar/<timestamp>/
```

输出：

| 文件 | 内容 |
|---|---|
| `sectors.csv` | 行业级别排名结果 |
| `stocks.csv` | 股票到行业的映射结果 |
| `unmatched_stocks.csv` | 未能映射行业的股票 |
| `sector_radar.md` | Markdown 报告 |

终端会打印：

```text
Sector report: ...
Stock mapping: ...
Unmatched stocks: ...
Markdown report: ...
Sectors ranked: ...; stocks mapped: ...; unmatched: ...
```

## 图 1：整体流程图

```text
┌──────────────────────┐
│ run_sector_radar.py  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ main()               │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 解析 CLI 参数         │
└──────────┬───────────┘
           │
           ▼
┌────────────────────────────┐
│ 读取 sector_radar.yaml      │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ 读取 strategy.yaml          │
│ 获取 cache_dir              │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ 初始化 TushareClient        │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ 读取股票列表                │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ build_sector_radar()        │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ 写出 CSV / Markdown 报告    │
└────────────────────────────┘
```

## 图 2：数据流结构

```text
inputs/sector_radar.smoke.csv
        │
        ▼
股票代码列表
        │
        ▼
Tushare: index_member_all
        │
        ▼
股票 → 申万一级行业
        │
        ▼
Tushare: sw_daily
        │
        ▼
行业历史行情
        │
        ▼
收益率 / 均线 / 波动率 / Score / Phase
        │
        ▼
outputs/sector_radar/<run_id>/
        ├── sectors.csv
        ├── stocks.csv
        ├── unmatched_stocks.csv
        └── sector_radar.md
```

## 图 3：Phase 判定树

```text
开始
 │
 ▼
close > MA20 > MA60
并且 5D收益 > A阈值
并且 20D收益 > A阈值？
 │
 ├── 是 ──► Phase A：强趋势赛道
 │
 └── 否
      │
      ▼
   close > MA60
   并且
   close <= MA20 或 5D收益 <= 0？
      │
      ├── 是 ──► Phase B：分化赛道
      │
      └── 否
           │
           ▼
        MA20 < MA60
        并且 20D收益 < C阈值？
           │
           ├── 是 ──► Phase C：失败赛道
           │
           └── 否 ──► Phase D：无结构赛道
```

## 图 4：输出关系

```text
一次运行目录
outputs/sector_radar/<timestamp>/
│
├── sectors.csv
│   └── 行业排名、Phase、Score、收益率、均线、包含股票
│
├── stocks.csv
│   └── 每只股票属于哪个行业、行业排名、Phase
│
├── unmatched_stocks.csv
│   └── 没有找到有效申万行业归属的股票
│
└── sector_radar.md
    └── 面向阅读的 Markdown 行业雷达报告
```
