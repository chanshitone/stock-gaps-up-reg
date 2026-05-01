from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def default_daily_win_loss_chart_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".html")


def _load_daily(csv_path: Path) -> pd.DataFrame:
    daily = pd.read_csv(csv_path)
    if daily.empty:
        raise ValueError(f"No rows found in {csv_path}")

    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    numeric_columns = [
        "daily_pnl",
        "equity",
        "cash_balance",
        "market_value",
        "daily_return_pct",
        "equity_trough",
        "raise",
        "raise_pct",
        "equity_peak",
        "pullback",
        "pullback_pct",
        "positions",
    ]
    for column in numeric_columns:
        if column in daily.columns:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")

    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def _build_title(daily: pd.DataFrame, csv_path: Path) -> str:
    start_equity = float(daily["equity"].iloc[0])
    end_equity = float(daily["equity"].iloc[-1])
    total_return_pct = ((end_equity / start_equity) - 1.0) * 100.0 if start_equity else 0.0
    max_pullback_pct = float(daily["pullback_pct"].min()) if "pullback_pct" in daily else 0.0
    return (
        f"{csv_path.stem} | End Equity ¥{end_equity:,.0f} | "
        f"Return {total_return_pct:.2f}% | Max Pullback {max_pullback_pct:.2f}%"
    )


def _pnl_bar_color(value: float) -> str:
    if value > 0:
        return "#2a9d8f"
    if value < 0:
        return "#d62828"
    return "#8d99ae"


def plot_daily_win_loss(csv_path: Path, output_path: Path, title: str | None = None) -> Path:
    daily = _load_daily(csv_path)
    plot_title = title or _build_title(daily, csv_path)

    dates = daily["date"]
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.3, 0.2],
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    if "equity_peak" in daily.columns:
        underwater = daily["equity_peak"] > daily["equity"]
        drawdown_top = daily["equity_peak"].where(underwater)
        drawdown_bottom = daily["equity"].where(underwater)
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown_top,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown_bottom,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(214, 40, 40, 0.12)",
                hoverinfo="skip",
                name="Drawdown",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=dates,
                y=daily["equity_peak"],
                mode="lines",
                name="Rolling peak",
                line={"color": "#9aa6b2", "width": 1.5, "dash": "dash"},
                hovertemplate="%{x|%Y-%m-%d}<br>Peak: ¥%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Scatter(
            x=dates,
            y=daily["equity"],
            mode="lines",
            name="Equity",
            line={"color": "#184e77", "width": 2.5},
            hovertemplate="%{x|%Y-%m-%d}<br>Equity: ¥%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    bar_colors = daily["daily_pnl"].apply(_pnl_bar_color)
    figure.add_trace(
        go.Bar(
            x=dates,
            y=daily["daily_pnl"],
            name="Daily P&L",
            marker_color=bar_colors.tolist(),
            hovertemplate="%{x|%Y-%m-%d}<br>P&L: ¥%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=dates,
            y=daily["positions"],
            mode="lines",
            name="Positions",
            line={"color": "#6a4c93", "width": 2.0, "shape": "hv"},
            fill="tozeroy",
            fillcolor="rgba(106, 76, 147, 0.12)",
            hovertemplate="%{x|%Y-%m-%d}<br>Positions: %{y}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=daily["daily_return_pct"],
            mode="lines",
            name="Daily return %",
            line={"color": "#ff9f1c", "width": 2.0},
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    figure.update_layout(
        title={"text": plot_title, "x": 0.5},
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        bargap=0.1,
        height=950,
        margin={"l": 70, "r": 70, "t": 90, "b": 60},
    )
    figure.update_xaxes(showgrid=False, rangeslider_visible=False)
    figure.update_yaxes(title_text="Equity (CNY)", row=1, col=1, gridcolor="rgba(0,0,0,0.08)")
    figure.update_yaxes(title_text="Daily P&L", row=2, col=1, zeroline=True, zerolinecolor="#6c757d", gridcolor="rgba(0,0,0,0.08)")
    figure.update_yaxes(title_text="Positions", row=3, col=1, secondary_y=False, gridcolor="rgba(0,0,0,0.08)")
    figure.update_yaxes(title_text="Return %", row=3, col=1, secondary_y=True, showgrid=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a daily win/loss equity CSV as an interactive HTML chart.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to *_daily_win_loss.csv")
    parser.add_argument("--output", type=Path, default=None, help="HTML output path (default: beside CSV)")
    parser.add_argument("--title", type=str, default=None, help="Optional chart title override")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    output_path = args.output.resolve() if args.output else default_daily_win_loss_chart_path(csv_path)
    written_path = plot_daily_win_loss(csv_path, output_path, args.title)
    print(f"Saved chart: {written_path}")


if __name__ == "__main__":
    main()