from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.stock_gaps_reg.daily_vs_indices_common import INDEX_SPECS


BUCKET_EDGES = [-np.inf, -2.0, -1.0, 0.0, 1.0, 2.0, np.inf]
BUCKET_LABELS = ["<=-2%", "-2%~-1%", "-1%~0%", "0%~1%", "1%~2%", ">=2%"]
REFERENCE_LINE_COLOR = "rgba(0,0,0,0.35)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot scatter and bucketed relationship charts between strategy daily returns and same-day index percentage changes."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to *_vs_indices.csv produced by run_compare_daily_win_loss_with_indices.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path (default: beside CSV)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional chart title override",
    )
    return parser.parse_args()


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".html")


def default_output_path_for_daily_csv(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_vs_indices.html")


def _marker_color(value: float) -> str:
    if value > 0:
        return "#2a9d8f"
    if value < 0:
        return "#d62828"
    return "#8d99ae"


def load_merged_view(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"No rows found in {csv_path}")

    required_columns = {"date", "strategy_return_pct"}
    required_columns.update({f"{slug}_pct_chg" for slug, _ts_code, _label in INDEX_SPECS})
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

    merged = frame.copy()
    merged["date"] = pd.to_datetime(merged["date"], errors="raise")
    numeric_columns = ["strategy_return_pct"] + [f"{slug}_pct_chg" for slug, _ts_code, _label in INDEX_SPECS]
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["strategy_result"] = merged.get("strategy_result", pd.Series(index=merged.index, dtype=object)).fillna("")
    merged["marker_color"] = merged["strategy_return_pct"].apply(_marker_color)
    return merged.sort_values("date").reset_index(drop=True)


def build_bucket_summary(merged: pd.DataFrame, index_col: str) -> pd.DataFrame:
    bucket_frame = merged[["strategy_return_pct", index_col]].dropna().copy()
    bucket_frame["bucket"] = pd.cut(
        bucket_frame[index_col],
        bins=BUCKET_EDGES,
        labels=BUCKET_LABELS,
        include_lowest=True,
        right=False,
    )
    summary = (
        bucket_frame.groupby("bucket", observed=False)
        .agg(
            avg_strategy_return_pct=("strategy_return_pct", "mean"),
            median_strategy_return_pct=("strategy_return_pct", "median"),
            strategy_win_rate_pct=("strategy_return_pct", lambda values: (values > 0).mean() * 100.0),
            days=("strategy_return_pct", "size"),
            avg_index_return_pct=(index_col, "mean"),
        )
        .reindex(BUCKET_LABELS)
        .reset_index()
    )
    return summary


def add_scatter_panel(
    figure: go.Figure,
    merged: pd.DataFrame,
    row: int,
    col: int,
    slug: str,
    label: str,
) -> None:
    index_col = f"{slug}_pct_chg"
    scatter_frame = merged[["date", "strategy_result", "strategy_return_pct", index_col, "marker_color"]].dropna()
    correlation = float(scatter_frame["strategy_return_pct"].corr(scatter_frame[index_col]))
    axis_number = (row - 1) * 3 + col
    xref = "x domain" if axis_number == 1 else f"x{axis_number} domain"
    yref = "y domain" if axis_number == 1 else f"y{axis_number} domain"

    figure.add_trace(
        go.Scatter(
            x=scatter_frame[index_col],
            y=scatter_frame["strategy_return_pct"],
            mode="markers",
            name=f"{label} scatter",
            showlegend=False,
            marker={"color": scatter_frame["marker_color"], "size": 8, "opacity": 0.75},
            customdata=np.column_stack(
                [
                    scatter_frame["date"].dt.strftime("%Y-%m-%d"),
                    scatter_frame["strategy_result"],
                ]
            ),
            hovertemplate=(
                "%{customdata[0]}<br>"
                + f"{label}: %{{x:.2f}}%<br>"
                + "策略日收益: %{y:.2f}%<br>"
                + "结果: %{customdata[1]}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )

    if len(scatter_frame) >= 2 and scatter_frame[index_col].nunique() >= 2:
        coefficients = np.polyfit(scatter_frame[index_col], scatter_frame["strategy_return_pct"], 1)
        x_line = np.linspace(scatter_frame[index_col].min(), scatter_frame[index_col].max(), 200)
        y_line = coefficients[0] * x_line + coefficients[1]
        figure.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"{label} fit",
                showlegend=False,
                line={"color": "#1d3557", "width": 2},
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    figure.add_hline(y=0.0, line_width=1, line_dash="dash", line_color=REFERENCE_LINE_COLOR, row=row, col=col)
    figure.add_vline(x=0.0, line_width=1, line_dash="dash", line_color=REFERENCE_LINE_COLOR, row=row, col=col)
    figure.add_annotation(
        x=0.02,
        y=0.98,
        xref=xref,
        yref=yref,
        text=f"corr = {correlation:.3f}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="rgba(0,0,0,0.12)",
        font={"size": 12, "color": "#1d3557"},
    )


def add_bucket_panel(
    figure: go.Figure,
    merged: pd.DataFrame,
    row: int,
    col: int,
    slug: str,
    label: str,
) -> None:
    index_col = f"{slug}_pct_chg"
    summary = build_bucket_summary(merged, index_col)

    figure.add_trace(
        go.Bar(
            x=summary["bucket"],
            y=summary["avg_strategy_return_pct"],
            name=f"{label} avg strategy return",
            showlegend=False,
            marker_color="#457b9d",
            text=summary["days"].apply(lambda value: f"n={int(value)}" if pd.notna(value) and value > 0 else ""),
            textposition="outside",
            customdata=np.column_stack(
                [
                    summary["median_strategy_return_pct"].fillna(np.nan),
                    summary["strategy_win_rate_pct"].fillna(np.nan),
                    summary["days"].fillna(0),
                    summary["avg_index_return_pct"].fillna(np.nan),
                ]
            ),
            hovertemplate=(
                "指数分桶: %{x}<br>"
                + "平均策略收益: %{y:.3f}%<br>"
                + "中位数策略收益: %{customdata[0]:.3f}%<br>"
                + "策略胜率: %{customdata[1]:.1f}%<br>"
                + "样本数: %{customdata[2]:.0f}<br>"
                + "该桶平均指数涨跌: %{customdata[3]:.3f}%<extra></extra>"
            ),
        ),
        row=row,
        col=col,
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=summary["bucket"],
            y=summary["strategy_win_rate_pct"],
            mode="lines+markers",
            name=f"{label} win rate",
            showlegend=False,
            line={"color": "#e76f51", "width": 2},
            marker={"size": 7},
            hovertemplate="指数分桶: %{x}<br>策略胜率: %{y:.1f}%<extra></extra>",
        ),
        row=row,
        col=col,
        secondary_y=True,
    )

    figure.add_hline(y=0.0, line_width=1, line_dash="dash", line_color=REFERENCE_LINE_COLOR, row=row, col=col, secondary_y=False)


def build_title(merged: pd.DataFrame, csv_path: Path) -> str:
    return (
        f"{csv_path.stem} | 策略日收益 vs 指数涨跌幅 | "
        f"{merged['date'].min():%Y-%m-%d} to {merged['date'].max():%Y-%m-%d}"
    )


def add_time_series_panel(figure: go.Figure, merged: pd.DataFrame) -> None:
    dates = merged["date"]
    palette = {
        "strategy": "#1d3557",
        "shanghai": "#457b9d",
        "shenzhen": "#2a9d8f",
        "chinext": "#e76f51",
    }

    figure.add_trace(
        go.Scatter(
            x=dates,
            y=merged["strategy_return_pct"],
            mode="lines+markers",
            name="策略日收益",
            line={"color": palette["strategy"], "width": 2.4},
            marker={"size": 4},
            hovertemplate="%{x|%Y-%m-%d}<br>策略日收益: %{y:.2f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )

    for slug, _ts_code, label in INDEX_SPECS:
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=merged[f"{slug}_pct_chg"],
                mode="lines",
                name=label,
                line={"color": palette[slug], "width": 1.8},
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: %{{y:.2f}}%<extra></extra>",
            ),
            row=3,
            col=1,
        )

    figure.add_hline(
        y=0.0,
        line_width=1,
        line_dash="dash",
        line_color=REFERENCE_LINE_COLOR,
        row=3,
        col=1,
    )


def plot_daily_vs_indices(csv_path: Path, output_path: Path, title: str | None = None) -> Path:
    merged = load_merged_view(csv_path)
    figure = make_subplots(
        rows=3,
        cols=3,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        subplot_titles=[label for _slug, _ts_code, label in INDEX_SPECS]
        + [f"{label} 分桶" for _slug, _ts_code, label in INDEX_SPECS]
        + ["阶段性同步和背离", None, None],
        row_heights=[0.34, 0.33, 0.33],
        specs=[
            [{}, {}, {}],
            [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}],
            [{"colspan": 3}, None, None],
        ],
    )

    for column_index, (slug, _ts_code, label) in enumerate(INDEX_SPECS, start=1):
        add_scatter_panel(figure, merged, 1, column_index, slug, label)
        add_bucket_panel(figure, merged, 2, column_index, slug, label)

        figure.update_xaxes(title_text=f"{label} 当日涨跌幅 (%)", row=1, col=column_index, zeroline=False)
        figure.update_yaxes(title_text="策略日收益 (%)", row=1, col=column_index, zeroline=False)
        figure.update_xaxes(title_text="指数涨跌幅分桶", row=2, col=column_index)
        figure.update_yaxes(title_text="平均策略日收益 (%)", row=2, col=column_index, secondary_y=False)
        figure.update_yaxes(title_text="策略胜率 (%)", row=2, col=column_index, secondary_y=True, range=[0, 100])

    add_time_series_panel(figure, merged)
    figure.update_xaxes(title_text="日期", row=3, col=1, showgrid=False)
    figure.update_yaxes(title_text="日涨跌幅 / 日收益 (%)", row=3, col=1)

    figure.update_layout(
        title={"text": title or build_title(merged, csv_path), "x": 0.5},
        template="plotly_white",
        hovermode="x unified",
        height=1320,
        margin={"l": 70, "r": 70, "t": 110, "b": 60},
        bargap=0.18,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def main() -> None:
    args = parse_args()

    csv_path = args.csv.resolve()
    output_path = args.output.resolve() if args.output else default_output_path(csv_path)
    written_path = plot_daily_vs_indices(csv_path, output_path, args.title)
    print(f"Saved chart: {written_path}")


if __name__ == "__main__":
    main()