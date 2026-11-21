"""
서울 일별 기온 시계열 분석 (2000~2024)

실행 순서
  1) python fetch_data.py   → data/seoul_weather_2000_2024.csv
  2) python analysis.py     → images/*.png, results/summary.json, REPORT.md

분석 질문
  Q1. 지난 25년간 서울 기온은 상승 추세인가? 상승했다면 얼마나 빠른가?
  Q2. 기온 변화는 계절(월)마다 같은가? 어느 달이 가장 많이 변했는가?
  Q3. 폭염일·열대야 같은 극단 기온일은 늘었는가?
  Q4. 하루 사이 기온 변동(노이즈)은 언제 가장 큰가?

적용한 시계열 기법
  - 이동평균(30일·365일): 노이즈를 걷어내고 계절성/추세를 분리해서 보기
  - 전일 대비 변화량: 기온은 0℃를 지나므로 %변화율 대신 ℃ 차이를 사용
  - 구간별 통계: 2000~2009년 vs 2015~2024년 월평균 비교
  - 변동성: 월별 전일 대비 변화량의 표준편차
  - 연도별 집계 + 선형 추세(최소제곱법)
  - (보너스) 시계열 분해, 베이스라인 예측
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "seoul_weather_2000_2024.csv"
IMG_DIR = ROOT / "images"
RESULT_DIR = ROOT / "results"
REPORT_PATH = ROOT / "REPORT.md"

HEAT_DAY = 33.0        # 폭염일: 일 최고기온 33℃ 이상 (기상청 폭염특보 기준 기온)
TROPICAL_NIGHT = 25.0  # 열대야: 일 최저기온 25℃ 이상 (기상청 정의는 18~09시 최저, 여기서는 일 최저로 근사)
COLD_DAY = -12.0       # 한파일: 일 최저기온 -12℃ 이하 (기상청 한파주의보 기준 기온)

PHYSICAL_RANGE = (-35.0, 45.0)   # 서울에서 물리적으로 가능한 기온 범위(이상치 1차 기준)
MAX_INTERP_GAP = 3               # 결측 3일 이하만 선형 보간
ANOMALY_Z = 4.0                  # 평년 대비 4σ 이상 벗어나면 '통계적 이상치'로 표시(삭제하지 않음)

PERIOD_1 = (2000, 2009)
PERIOD_2 = (2015, 2024)
FORECAST_HORIZON = 30

MONTH_KO = [f"{m}월" for m in range(1, 13)]
SEASON_OF_MONTH = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
                   6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}

COLORS = {"main": "#2F6DB5", "accent": "#D9534F", "soft": "#9DB9DD", "gray": "#8A8A8A",
          "warm": "#E8743B", "cool": "#3B8BC2", "green": "#3E9C5F"}


def setup_font() -> None:
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
                 "Noto Sans CJK JP"]:
        if name in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


# ─────────────────────────────────────────────
# 1. 데이터 불러오기 / 기본 정보
# ─────────────────────────────────────────────
def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. 먼저 `python fetch_data.py`를 실행하세요.")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def inspect(df: pd.DataFrame) -> dict:
    dates = df["date"]
    full_range = pd.date_range(dates.min(), dates.max(), freq="D")
    return {
        "rows": int(len(df)),
        "start": dates.min().strftime("%Y-%m-%d"),
        "end": dates.max().strftime("%Y-%m-%d"),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_dates": int(dates.duplicated().sum()),
        "missing_dates": int(len(full_range.difference(dates))),
    }


# ─────────────────────────────────────────────
# 2. 정제 (결측치 / 이상치)
# ─────────────────────────────────────────────
def day_of_year(index: pd.DatetimeIndex) -> np.ndarray:
    """윤년의 2/29 이후를 하루 당겨 모든 해를 365일 기준으로 맞춘다."""
    doy = index.dayofyear.to_numpy().copy()
    leap_after_feb = index.is_leap_year & (doy > 59)
    doy[leap_after_feb] -= 1
    return doy


def climatology(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """날짜(1~365)별 평년값과 표준편차. 15일 원형 이동평균으로 부드럽게 만든다."""
    doy = day_of_year(series.index)
    g = series.groupby(doy)
    mean, std = g.mean(), g.std()

    def circular_smooth(s: pd.Series, w: int = 15) -> pd.Series:
        s = s.reindex(range(1, 366)).interpolate(limit_direction="both")
        ext = pd.concat([s.iloc[-w:], s, s.iloc[:w]])
        sm = ext.rolling(w, center=True, min_periods=1).mean()
        return sm.iloc[w:-w].set_axis(s.index)

    return circular_smooth(mean), circular_smooth(std)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log: dict = {}
    df = raw.copy()

    # (1) 중복 날짜 제거
    log["duplicates_removed"] = int(df["date"].duplicated().sum())
    df = df.drop_duplicates("date", keep="first").set_index("date").sort_index()

    # (2) 빠진 날짜를 채워 '매일 1행' 시계열로 만든다 (시간 축 정렬)
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    log["dates_added"] = int(len(full.difference(df.index)))
    df = df.reindex(full)
    df.index.name = "date"

    temp_cols = [c for c in ["tmax", "tmin", "tmean"] if c in df.columns]

    # (3) 물리적으로 불가능한 값 → 결측 처리
    lo, hi = PHYSICAL_RANGE
    out_of_range = {}
    for c in temp_cols:
        mask = (df[c] < lo) | (df[c] > hi)
        out_of_range[c] = int(mask.sum())
        df.loc[mask, c] = np.nan
    if "precip" in df.columns:
        neg = df["precip"] < 0
        out_of_range["precip"] = int(neg.sum())
        df.loc[neg, "precip"] = np.nan
    log["out_of_range_to_nan"] = out_of_range

    # (4) 논리 오류: 최저기온 > 최고기온 → 둘 다 결측
    if {"tmax", "tmin"} <= set(df.columns):
        bad = df["tmin"] > df["tmax"]
        log["tmin_gt_tmax"] = int(bad.sum())
        df.loc[bad, ["tmax", "tmin"]] = np.nan

    # (5) 평균기온이 없으면 (최고+최저)/2 로 계산
    if "tmean" not in df.columns:
        df["tmean"] = (df["tmax"] + df["tmin"]) / 2
        log["tmean_computed"] = "전체 (API에 평균기온 없음)"
        temp_cols.append("tmean")
    else:
        fill = df["tmean"].isna() & df["tmax"].notna() & df["tmin"].notna()
        df.loc[fill, "tmean"] = (df.loc[fill, "tmax"] + df.loc[fill, "tmin"]) / 2
        log["tmean_computed"] = int(fill.sum())

    # (6) 결측 보간: 연속 3일 이하만 선형 보간, 더 긴 공백은 그대로 둔다
    before = {c: int(df[c].isna().sum()) for c in temp_cols}
    for c in temp_cols:
        # pandas의 limit은 긴 공백의 앞부분도 채우므로, 공백 길이를 직접 재서 짧은 공백만 채운다
        isna = df[c].isna()
        gap_id = (isna != isna.shift()).cumsum()
        gap_len = isna.groupby(gap_id).transform("sum")
        short_gap = isna & (gap_len <= MAX_INTERP_GAP)
        filled = df[c].interpolate(method="time", limit_area="inside")
        df.loc[short_gap, c] = filled[short_gap]
    after = {c: int(df[c].isna().sum()) for c in temp_cols}
    log["interpolated"] = {c: before[c] - after[c] for c in temp_cols}
    log["remaining_missing"] = after
    if "precip" in df.columns:
        log["precip_missing_kept"] = int(df["precip"].isna().sum())  # 강수는 보간하지 않음

    # (7) 통계적 이상치: 평년 대비 |z| ≥ 4 → 표시만 하고 유지 (실제 극한 기상일 수 있음)
    clim_mean, clim_std = climatology(df["tmean"])
    doy = day_of_year(df.index)
    df["clim"] = clim_mean.reindex(doy).to_numpy()
    df["anomaly"] = df["tmean"] - df["clim"]
    z = df["anomaly"] / clim_std.reindex(doy).to_numpy()
    df["is_outlier"] = z.abs() >= ANOMALY_Z
    log["statistical_outliers_flagged"] = int(df["is_outlier"].sum())
    log["statistical_outlier_dates"] = [d.strftime("%Y-%m-%d") for d in df.index[df["is_outlier"]]][:10]
    return df, log


# ─────────────────────────────────────────────
# 3. 시계열 특성 추가
# ─────────────────────────────────────────────
def add_features(df: pd.DataFrame, short: int = 30, long: int = 365) -> pd.DataFrame:
    df = df.copy()
    # 창이 다 차기 전(시작 부분)의 평균은 일부 계절만 반영해 왜곡되므로 비워 둔다.
    # 창 안에 결측이 조금(10% 이하) 있는 것은 허용한다.
    for col, w in (("ma_short", short), ("ma_long", long)):
        ma = df["tmean"].rolling(w, min_periods=int(w * 0.9)).mean()
        ma.iloc[: w - 1] = np.nan
        df[col] = ma
    df["change"] = df["tmean"].diff()               # 전일 대비 변화량(℃)
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["heat_day"] = df["tmax"] >= HEAT_DAY
    df["tropical_night"] = df["tmin"] >= TROPICAL_NIGHT
    df["cold_day"] = df["tmin"] <= COLD_DAY
    return df


def linear_trend(x: np.ndarray, y: np.ndarray) -> dict:
    """최소제곱 직선 y = a·x + b. 기울기 표준오차로 대략적인 95% 구간을 계산한다."""
    mask = ~np.isnan(y)
    x, y = x[mask].astype(float), y[mask]
    n = len(x)
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    ss_res, ss_tot = float((resid ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    se = math.sqrt(ss_res / (n - 2) / ((x - x.mean()) ** 2).sum()) if n > 2 else float("nan")
    return {"slope": float(a), "intercept": float(b), "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
            "se": se, "ci95": (float(a - 2 * se), float(a + 2 * se)), "n": n}


def annual_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("year")
    out = pd.DataFrame({
        "valid_days": g["tmean"].count(),
        "tmean": g["tmean"].mean(),
        "tmax_mean": g["tmax"].mean(),
        "tmin_mean": g["tmin"].mean(),
        "heat_days": g["heat_day"].sum(),
        "tropical_nights": g["tropical_night"].sum(),
        "cold_days": g["cold_day"].sum(),
    })
    return out[out["valid_days"] >= 360]  # 거의 완전한 해만 사용 (부분 연도는 평균을 왜곡)


def period_monthly(df: pd.DataFrame, p1=PERIOD_1, p2=PERIOD_2) -> pd.DataFrame:
    def mm(p):
        sub = df[(df["year"] >= p[0]) & (df["year"] <= p[1])]
        return sub.groupby("month")["tmean"].mean()

    out = pd.DataFrame({"p1": mm(p1), "p2": mm(p2)}).reindex(range(1, 13))
    out["diff"] = out["p2"] - out["p1"]
    return out


def monthly_volatility(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("month")["change"]
    return pd.DataFrame({"std": g.std(), "mean_abs": g.apply(lambda s: s.abs().mean()),
                         "max_drop": g.min(), "max_rise": g.max()}).reindex(range(1, 13))


# ─────────────────────────────────────────────
# 4. (보너스) 분해 / 베이스라인 예측
# ─────────────────────────────────────────────
def decompose(df: pd.DataFrame) -> pd.DataFrame:
    """고전적 가법 분해: 관측값 = 추세 + 계절성 + 잔차"""
    s = df["tmean"]
    trend = s.rolling(365, center=True, min_periods=355).mean()
    trend.iloc[:182] = np.nan    # 양 끝 반년은 창이 다 차지 않아 계절 편향이 생기므로 제외
    trend.iloc[-182:] = np.nan
    detrended = s - trend
    doy = day_of_year(s.index)
    seasonal_by_doy = detrended.groupby(doy).mean()
    seasonal_by_doy = seasonal_by_doy - seasonal_by_doy.mean()
    ext = pd.concat([seasonal_by_doy.iloc[-15:], seasonal_by_doy, seasonal_by_doy.iloc[:15]])
    seasonal_by_doy = ext.rolling(15, center=True, min_periods=1).mean().iloc[15:-15].set_axis(seasonal_by_doy.index)
    seasonal = pd.Series(seasonal_by_doy.reindex(doy).to_numpy(), index=s.index)
    return pd.DataFrame({"observed": s, "trend": trend, "seasonal": seasonal, "resid": s - trend - seasonal})


def baseline_forecast(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> tuple[pd.DataFrame, dict]:
    """마지막 horizon일을 숨기고 네 가지 단순 방법으로 맞혀 본다."""
    s = df["tmean"].dropna()
    train, test = s.iloc[:-horizon], s.iloc[-horizon:]
    clim_mean, _ = climatology(train)
    test_doy = day_of_year(test.index)
    clim = clim_mean.reindex(test_doy).to_numpy()
    recent_anom = float((train.iloc[-7:] - clim_mean.reindex(day_of_year(train.index[-7:])).to_numpy()).mean())

    preds = pd.DataFrame(index=test.index)
    preds["actual"] = test
    preds["naive"] = train.iloc[-1]                                           # 마지막 값 유지
    preds["seasonal_naive"] = s.reindex(test.index - pd.DateOffset(years=1)).to_numpy()  # 작년 같은 날
    preds["climatology"] = clim                                               # 평년값
    preds["clim_plus_anomaly"] = clim + recent_anom                           # 평년값 + 최근 7일 편차

    mae = {m: float((preds[m] - preds["actual"]).abs().mean())
           for m in ["naive", "seasonal_naive", "climatology", "clim_plus_anomaly"]}
    return preds, {"mae": mae, "recent_anomaly": recent_anom, "train_end": train.index[-1].strftime("%Y-%m-%d"),
                   "test_start": test.index[0].strftime("%Y-%m-%d"), "test_end": test.index[-1].strftime("%Y-%m-%d")}


# ─────────────────────────────────────────────
# 5. 시각화
# ─────────────────────────────────────────────
def save(fig, name: str) -> str:
    IMG_DIR.mkdir(exist_ok=True)
    path = IMG_DIR / name
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return f"images/{name}"


def plot_daily(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(df.index, df["tmean"], color=COLORS["soft"], lw=0.4, alpha=0.7, label="일평균 기온 (원자료)")
    ax.plot(df.index, df["ma_short"], color=COLORS["main"], lw=1.0, label="30일 이동평균 (계절성)")
    ax.plot(df.index, df["ma_long"], color=COLORS["accent"], lw=2.2, label="365일 이동평균 (추세)")
    ax.set_title("서울 일평균 기온과 이동평균 (2000–2024)", fontsize=14, loc="left", fontweight="bold")
    ax.set_ylabel("기온 (℃)")
    ax.legend(loc="lower left", ncol=3, frameon=False, fontsize=9)
    return save(fig, "01_daily_temperature_moving_average.png")


def plot_annual_trend(annual: pd.DataFrame, trend: dict) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = annual["tmean"]
    base = y.mean()
    colors = [COLORS["warm"] if v >= base else COLORS["cool"] for v in y]
    ax.bar(annual.index, y - base, bottom=base, color=colors, width=0.75, alpha=0.85)
    xs = np.array(annual.index, dtype=float)
    ax.plot(xs, trend["slope"] * xs + trend["intercept"], color="#222", lw=2, ls="--",
            label=f"선형 추세 {trend['slope'] * 10:+.2f}℃/10년 (R²={trend['r2']:.2f})")
    ax.axhline(base, color=COLORS["gray"], lw=0.8)
    ax.text(annual.index[0] - 0.4, base, f" 25년 평균 {base:.1f}℃", va="bottom", fontsize=9, color=COLORS["gray"])
    top = y.idxmax()
    ax.annotate(f"{top}년 {y.max():.2f}℃", (top, y.max()), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(y.min() - 0.4, y.max() + 0.4)
    ax.set_title("연평균 기온과 선형 추세", fontsize=14, loc="left", fontweight="bold")
    ax.set_ylabel("연평균 기온 (℃)")
    ax.legend(frameon=False, loc="upper left")
    return save(fig, "02_annual_mean_trend.png")


def plot_monthly_compare(pm: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(1, 13)
    ax1.plot(x, pm["p1"], marker="o", color=COLORS["cool"], label=f"{PERIOD_1[0]}–{PERIOD_1[1]}")
    ax1.plot(x, pm["p2"], marker="o", color=COLORS["warm"], label=f"{PERIOD_2[0]}–{PERIOD_2[1]}")
    ax1.set_xticks(x, MONTH_KO)
    ax1.set_ylabel("월평균 기온 (℃)")
    ax1.set_title("월별 평균 기온: 두 구간 비교", fontsize=13, loc="left", fontweight="bold")
    ax1.legend(frameon=False)

    colors = [COLORS["warm"] if d >= 0 else COLORS["cool"] for d in pm["diff"]]
    ax2.bar(x, pm["diff"], color=colors)
    for xi, d in zip(x, pm["diff"]):
        ax2.text(xi, d + (0.03 if d >= 0 else -0.03), f"{d:+.1f}", ha="center",
                 va="bottom" if d >= 0 else "top", fontsize=8)
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_xticks(x, [str(m) for m in x])
    ax2.set_xlabel("월")
    ax2.set_ylabel("기온 차이 (℃)")
    ax2.set_title("최근 구간 − 과거 구간", fontsize=13, loc="left", fontweight="bold")
    return save(fig, "03_monthly_period_comparison.png")


def plot_extremes(annual: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.27
    x = np.array(annual.index)
    ax.bar(x - w, annual["heat_days"], w, color=COLORS["accent"], label=f"폭염일 (최고 ≥{HEAT_DAY:.0f}℃)")
    ax.bar(x, annual["tropical_nights"], w, color=COLORS["warm"], label=f"열대야 (최저 ≥{TROPICAL_NIGHT:.0f}℃)")
    ax.bar(x + w, annual["cold_days"], w, color=COLORS["cool"], label=f"한파일 (최저 ≤{COLD_DAY:.0f}℃)")
    ax.set_title("연도별 극단 기온일 수", fontsize=14, loc="left", fontweight="bold")
    ax.set_ylabel("일수")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    return save(fig, "04_extreme_days_per_year.png")


def plot_volatility(df: pd.DataFrame, vol: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    data = [df.loc[df["month"] == m, "change"].dropna() for m in range(1, 13)]
    ax1.boxplot(data, showfliers=True,
                flierprops={"markersize": 2, "alpha": 0.3}, medianprops={"color": COLORS["accent"]})
    ax1.set_xticks(range(1, 13), [str(m) for m in range(1, 13)])
    ax1.axhline(0, color="#333", lw=0.6)
    ax1.set_xlabel("월")
    ax1.set_ylabel("전일 대비 변화량 (℃)")
    ax1.set_title("월별 전일 대비 기온 변화 분포", fontsize=13, loc="left", fontweight="bold")

    top = vol["std"].idxmax()
    colors = [COLORS["accent"] if m == top else COLORS["main"] for m in range(1, 13)]
    ax2.bar(range(1, 13), vol["std"], color=colors)
    ax2.set_xticks(range(1, 13))
    ax2.set_xlabel("월")
    ax2.set_ylabel("표준편차 (℃)")
    ax2.set_title("월별 변동성 (변화량 표준편차)", fontsize=13, loc="left", fontweight="bold")
    return save(fig, "05_daily_change_volatility.png")


def plot_decomposition(dec: pd.DataFrame) -> str:
    fig, axes = plt.subplots(4, 1, figsize=(12, 8.5), sharex=True)
    specs = [("observed", "관측값", COLORS["soft"]), ("trend", "추세 (365일 중심 이동평균)", COLORS["accent"]),
             ("seasonal", "계절성 (날짜별 평균 패턴)", COLORS["main"]), ("resid", "잔차 (노이즈)", COLORS["gray"])]
    for ax, (col, title, c) in zip(axes, specs):
        ax.plot(dec.index, dec[col], color=c, lw=0.6 if col in ("observed", "resid") else 1.5)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("℃")
    axes[0].text(0, 1.25, "시계열 분해: 관측값 = 추세 + 계절성 + 잔차", transform=axes[0].transAxes,
                 fontsize=14, fontweight="bold")
    return save(fig, "06_decomposition.png")


def plot_forecast(df: pd.DataFrame, preds: pd.DataFrame, fc: dict) -> str:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    hist = df["tmean"].loc[preds.index[0] - pd.Timedelta(days=60): preds.index[0] - pd.Timedelta(days=1)]
    ax.plot(hist.index, hist, color=COLORS["gray"], lw=1.2, label="학습 구간 (실제)")
    ax.plot(preds.index, preds["actual"], color="#222", lw=2, label="실제값 (정답)")
    labels = {"naive": "마지막 값 유지", "seasonal_naive": "작년 같은 날", "climatology": "평년값",
              "clim_plus_anomaly": "평년값+최근 편차"}
    styles = {"naive": (COLORS["gray"], ":"), "seasonal_naive": (COLORS["green"], "-."),
              "climatology": (COLORS["main"], "--"), "clim_plus_anomaly": (COLORS["accent"], "-")}
    for m, lab in labels.items():
        c, ls = styles[m]
        ax.plot(preds.index, preds[m], color=c, ls=ls, lw=1.4, label=f"{lab} (MAE {fc['mae'][m]:.2f}℃)")
    ax.axvline(preds.index[0], color="#999", lw=0.8)
    ax.set_title(f"베이스라인 예측: 마지막 {len(preds)}일 맞혀 보기", fontsize=14, loc="left", fontweight="bold")
    ax.set_ylabel("일평균 기온 (℃)")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
    return save(fig, "07_baseline_forecast.png")


# ─────────────────────────────────────────────
# 6. 리포트 작성
# ─────────────────────────────────────────────
def kv(d: dict) -> str:
    """{'tmax': 1, 'tmin': 0} → 'tmax 1, tmin 0'"""
    return ", ".join(f"{k} {v}" for k, v in d.items())


SEASON_HYPOTHESIS = {
    "여름": "북태평양 고기압이 더 강하거나 오래 머무는 여름이 잦아졌을 가능성, 그리고 열대야가 늘면서 밤 기온이 떨어지지 않는 효과",
    "가을": "여름 더위가 9~10월까지 길어지는 '여름 연장' 현상",
    "겨울": "겨울은 해마다 기복이 커서(한파 해 vs 따뜻한 해) 10년 평균 차이에 우연이 섞였을 가능성이 크다. 북극 진동 같은 대규모 순환의 영향",
    "봄": "봄철 이동성 고기압이 잦아지고 꽃 피는 시기가 빨라지는 것과 같은 방향의 변화",
}

VOLATILITY_HYPOTHESIS = {
    "겨울": "시베리아 고기압이 확장했다 약해지기를 반복하면서 한파가 들어왔다 빠지는 '삼한사온'형 변동",
    "봄": "이동성 고기압과 저기압이 빠르게 지나가며 맑은 날과 흐린 날이 번갈아 나타나는 봄철 날씨 특성",
    "가을": "찬 대륙 공기가 처음 내려오며 기온이 계단식으로 떨어지는 환절기 특성",
    "여름": "장마·소나기에 따른 일시적 기온 하락",
}


def build_report(info: dict, log: dict, annual: pd.DataFrame, trend: dict, pm: pd.DataFrame,
                 vol: pd.DataFrame, dec: pd.DataFrame, fc: dict, imgs: dict, meta: dict) -> str:
    # ── 핵심 수치 ──
    first5, last5 = annual["tmean"].iloc[:5], annual["tmean"].iloc[-5:]
    warm_years = annual["tmean"].sort_values(ascending=False).head(3)
    cold_year = annual["tmean"].idxmin()
    per_decade = trend["slope"] * 10
    lo, hi = (v * 10 for v in trend["ci95"])
    trend_significant = lo > 0 or hi < 0

    max_m, min_m = int(pm["diff"].idxmax()), int(pm["diff"].idxmin())
    season_max = SEASON_OF_MONTH[max_m]
    n_warmer_months = int((pm["diff"] > 0).sum())

    p1_mask = (annual.index >= PERIOD_1[0]) & (annual.index <= PERIOD_1[1])
    p2_mask = (annual.index >= PERIOD_2[0]) & (annual.index <= PERIOD_2[1])
    heat1, heat2 = annual.loc[p1_mask, "heat_days"].mean(), annual.loc[p2_mask, "heat_days"].mean()
    trop1, trop2 = annual.loc[p1_mask, "tropical_nights"].mean(), annual.loc[p2_mask, "tropical_nights"].mean()
    cold1, cold2 = annual.loc[p1_mask, "cold_days"].mean(), annual.loc[p2_mask, "cold_days"].mean()
    heat_top = annual["heat_days"].idxmax()
    trop_top = annual["tropical_nights"].idxmax()

    vmax_m, vmin_m = int(vol["std"].idxmax()), int(vol["std"].idxmin())
    seasonal_amp = dec["seasonal"].max() - dec["seasonal"].min()
    resid_std = dec["resid"].std()
    trend_range = dec["trend"].max() - dec["trend"].min()
    best_model = min(fc["mae"], key=fc["mae"].get)
    model_ko = {"naive": "마지막 값 유지", "seasonal_naive": "작년 같은 날", "climatology": "평년값",
                "clim_plus_anomaly": "평년값 + 최근 7일 편차"}

    if trend_significant and per_decade > 0:
        insight1_title = f"서울은 10년에 약 {per_decade:.1f}℃씩 따뜻해지고 있다"
    elif trend_significant:
        insight1_title = f"서울은 10년에 약 {abs(per_decade):.1f}℃씩 서늘해지고 있다"
    else:
        insight1_title = f"25년 자료로는 뚜렷한 기온 추세가 확인되지 않는다 ({per_decade:+.2f}℃/10년)"

    trend_word = "상승" if per_decade > 0 else "하락"
    if trend_significant:
        q1_conclusion = (f"서울의 연평균 기온은 {per_decade:+.2f}℃/10년으로 {trend_word}했다. "
                         "365일 이동평균과 연평균 그래프 모두 같은 방향을 보인다.")
    else:
        q1_conclusion = (f"연평균 기온 추세는 {per_decade:+.2f}℃/10년이지만 해마다 기복이 커서, "
                         "25년 자료만으로는 상승·하락을 단정할 수 없다.")

    heat_up = heat2 > heat1
    if heat_up:
        insight3_title = "평균이 조금 오르면 극단값은 훨씬 크게 늘어난다"
        insight3_why = ("기온 분포 전체가 오른쪽으로 조금만 이동해도, 분포 끝에 있는 '33℃ 이상' 같은 기준을 넘는 날의 "
                        "비율은 크게 늘어난다. 평균 변화(인사이트 1)는 1℃ 안팎인데 극단 일수 변화가 더 크게 보이는 이유다.")
    else:
        insight3_title = "평균 기온 변화와 달리 폭염일은 뚜렷하게 늘지 않았다"
        insight3_why = ("평균이 올라도 여름 최고기온 분포는 장마·강수일 변화에 따라 달라질 수 있다. "
                        "또 격자 자료는 극한 최고기온을 낮게 잡는 경향이 있어 33℃ 기준을 넘는 날이 적게 집계됐을 수 있다.")

    def ratio_text(a: float, b: float) -> str:
        if a == 0:
            return "과거 구간에는 거의 없었다가 새로 나타남" if b > 0 else "두 구간 모두 거의 없음"
        return f"{b / a:.1f}배"

    ci_text = (f"대략적인 95% 범위 {lo:+.2f}~{hi:+.2f}℃로 0을 포함하지 않아, 우연한 흔들림만으로 보기 어렵다"
               if trend_significant else
               f"대략적인 95% 범위 {lo:+.2f}~{hi:+.2f}℃로 0을 포함해, 25년 자료만으로는 추세라고 단정하기 어렵다")

    missing_lines = "\n".join(f"| `{c}` | {info['dtypes'][c]} | {info['missing'][c]} |" for c in info["columns"])
    outlier_dates = ", ".join(log["statistical_outlier_dates"]) or "없음"

    mae_rows = "\n".join(f"| {model_ko[m]} | {fc['mae'][m]:.2f} |" for m in
                         sorted(fc["mae"], key=fc["mae"].get))

    return f"""# 서울 25년(2000~2024) 기온 트렌드 분석 리포트

> 이 리포트의 수치와 그래프는 `analysis.py`가 데이터에서 직접 계산해 생성했습니다.
> **먼저 읽어 주세요.** 이 문서는 통계학을 배우지 않은 사람도 이해할 수 있도록, 용어의 뜻과 분석 방법을 선택한 이유를 함께 적은 보고서입니다.
>
> - **관찰(Fact)**: 데이터나 그래프에서 직접 계산해 확인한 수치입니다.
> - **원인 가설(Why)**: 그 수치가 나타난 이유에 대한 *가능한 설명*입니다. 이번 분석만으로 원인을 증명하지 못합니다.
> - **다음 확인(Action)**: 가설이 맞는지 확인하거나 결과를 활용하기 위해 할 일입니다.
>
> 수치·표·그래프는 `analysis.py`가 실제 CSV에서 계산해 다시 만듭니다. 원인 가설과 AI 사용 로그는 제출자가 직접 검토해야 합니다.

## 1. 분석 주제 및 선정 이유 — 왜 서울 기온을 알아볼까요?

**분석 질문은 “서울의 기온이 2000년부터 2024년까지 25년 동안 어떻게 달라졌는가?”입니다.**

“요즘은 여름이 더 덥고 밤에도 잠들기 어렵다”는 이야기를 자주 듣습니다. 하지만 작년에 유난히 더운 날이 있었다는 기억만으로 **25년 동안 실제 기온이 계속 올라갔는지**는 알 수 없습니다. 그래서 날짜별 기온 기록을 모아 숫자로 비교했습니다.

여기서 중요한 말 세 가지를 먼저 알아두면 그래프가 훨씬 쉬워집니다.

| 용어 | 일상적인 뜻 | 왜 구분하나요? |
|---|---|---|
| **추세(Trend)** | 여러 해에 걸쳐 전체적으로 조금씩 올라가거나 내려가는 방향 | 서울이 장기적으로 따뜻해졌는지 보기 위해 |
| **계절성(Seasonality)** | 여름은 덥고 겨울은 추운 것처럼 해마다 반복되는 변화 | 계절 때문에 생기는 큰 변화와 장기 추세를 구분하기 위해 |
| **노이즈(Noise)** | 어느 날 갑자기 춥거나 더워지는 불규칙한 변화 | 날씨가 하루하루 흔들려도 장기 추세를 놓치지 않기 위해 |

**시계열 데이터**는 이렇게 *날짜 순서대로 쌓인 자료*입니다. 같은 수치라도 언제 기록됐는지가 중요합니다. 이 프로젝트는 **Open-Meteo**라는 날씨 데이터 제공 서비스에서 자료를 받아 왔고, **API**(프로그램이 데이터를 자동으로 받아오는 통로)를 이용해 같은 조건의 자료를 다시 가져올 수 있게 했습니다. 다만 이 자료는 서울 한 관측소에서 직접 측정한 실측값이 아니라 **ERA5 재분석 격자 자료**라는 점을 구분해야 합니다.

## 2. 분석 질문 — 무엇을, 왜, 어떻게 확인하나요?

| 질문 | 쉬운 설명 | 분석 방법은 무엇인가요? | 왜 이 방법을 쓰나요? |
|---|---|---|---|
| **Q1. 25년 동안 따뜻해졌을까요?** | 하루가 아니라 여러 해를 보면 서울의 평균 기온이 올라갔는가? | **365일 이동평균**은 앞뒤 약 1년의 값을 평균 내 큰 계절 변화를 완화합니다. **연평균**은 한 해 기온을 하나의 숫자로 요약하고, **선형 추세**는 연도별 점들을 가장 잘 대표하는 직선입니다. | 매일의 폭염·한파만 보면 전체 흐름을 놓치므로, 장기 변화의 방향과 10년당 변화량을 확인하기 위해서입니다. |
| **Q2. 모든 달이 똑같이 변했을까요?** | 1월과 8월, 3월 중 어느 달의 기온 차이가 가장 클까요? | **구간별 월평균 비교**는 2000~2009년과 2015~2024년의 같은 달끼리 평균 기온을 비교하는 것입니다. | 연평균이 올라도 어느 계절이 가장 달라졌는지는 알 수 없기 때문입니다. |
| **Q3. 아주 더운 날과 밤도 늘었을까요?** | 평균 기온 말고 33℃ 이상인 날이나 25℃ 이상인 밤에 가까운 날은 얼마나 늘었을까요? | **기준 초과 일수 집계**는 정해진 기온 기준에 해당하는 날을 해마다 세는 방법입니다. | 평균이 조금 변해도 생활에 불편한 고온일 수는 크게 달라질 수 있기 때문입니다. 이 분석의 '열대야'는 *일 최저기온 기준의 근사치*입니다. |
| **Q4. 어느 달에 기온이 크게 흔들릴까요?** | 어제와 오늘의 기온 차이는 어느 달에 가장 클까요? | **전일 대비 변화량**은 '오늘 평균기온 − 어제 평균기온'이고, **표준편차**는 그 차이가 얼마나 들쭉날쭉한지 보여 주는 숫자입니다. | 같은 평균 기온이어도 어떤 달은 날마다 급변할 수 있기 때문입니다. |

**예를 들어 이동평균은** 어제·오늘·내일의 기온이 10℃, 12℃, 14℃라면 세 날 평균인 12℃로 큰 흐름을 보는 것과 비슷합니다. 실제 분석에서는 30일과 365일을 사용했습니다. **표준편차는** 값이 일정하면 작고 값이 크게 오르내리면 커집니다. 그래서 평균만으로 알 수 없는 '흔들림'을 비교하는 데 쓰입니다.

## 3. 데이터 설명 — 어떤 자료를 사용했나요?

**이 자료는 서울시청 부근의 날씨를 하루에 한 줄씩 정리한 표입니다.** 서울 한 장소에 기온계를 25년 동안 설치해 직접 읽은 자료가 아니라, 관측값과 기상 모델 등을 결합해 과거 날씨를 재구성한 **ERA5 재분석 자료**입니다. 이를 **격자**(지도를 나눈 칸) 형태로 제공하기 때문에 관측소의 실측과는 값이 다를 수 있습니다.

| 항목 | 내용 |
|---|---|
| 출처 | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (ECMWF의 ERA5 재분석 자료를 제공하는 서비스) |
| 지점 | 서울시청 부근 (위도 37.5665, 경도 126.9780), 격자 해발고도 {meta.get('elevation_m', '-')} m |
| 기간 | {info['start']} ~ {info['end']} |
| 데이터 수 | **{info['rows']:,}일** (날짜 하나마다 한 행이며 윤년의 2월 29일도 포함) |
| 사용 모델 | {meta.get('request', {}).get('model', '-')} (전 기간을 가능한 한 같은 모델·조건으로 비교하기 위해 선택) |
| 라이선스 | CC BY 4.0 — Weather data by Open-Meteo.com, Generated using Copernicus Climate Change Service information |
| 수집 방법 | `python fetch_data.py` (요청 URL은 `data/metadata.json`에 기록) |

**데이터 열(컬럼)과 빠진 값(결측치) 읽는 법**

**컬럼(Column)**은 표의 세로 항목, **행(Row)**은 날짜별 한 줄입니다. **결측치**는 그 날짜에 기록이 비어 있는 칸을 뜻합니다. 아래의 '결측 수 0'은 *현재 저장된 분석 자료에 비어 있는 값이 없다*는 뜻이지, 모든 기상 데이터에 결측이 없다는 뜻은 아닙니다.

| 열 이름 | 쉬운 뜻 | 단위 | 분석에 쓰는 이유 |
|---|---|---|---|
| `date` | 기온을 기록한 날짜 | 날짜 | 날짜순으로 정렬하고 연·월 단위로 묶으려고 |
| `tmax` | 하루 중 가장 높은 기온 | ℃ | 33℃ 이상인 더운 날을 세려고 |
| `tmin` | 하루 중 가장 낮은 기온 | ℃ | 더운 밤에 가까운 날이나 추운 날을 세려고 |
| `tmean` | 하루 평균 기온 | ℃ | 장기 추세와 하루 사이 변화량을 분석하려고 |
| `precip` | 하루 동안의 강수량 | mm | 기초 자료로 보관했으며, 이번 주요 분석은 기온에 집중함 |

| 컬럼 | 컴퓨터에서 저장한 자료형 | 결측 수 |
|---|---|---|
{missing_lines}

**자료형은 무슨 뜻인가요?** `datetime64[us]`는 날짜·시간으로 인식한 값, `float64`는 소수점이 있는 숫자를 저장하는 형식입니다. **`float64`라고 해서 기온을 소수점 아래 64자리까지 측정했다는 뜻은 아닙니다.**

**API 키**는 서비스를 이용할 때 쓰는 인증용 비밀번호와 비슷한 값인데, 이 프로젝트의 데이터 요청은 별도 키 없이 실행하도록 구성되어 있습니다. `fetch_data.py`가 자료를 수집하고 `data/metadata.json`에는 실제 요청 조건을 남깁니다. **재현**이란 다른 사람이 같은 자료·코드·조건으로 계산을 다시 해보고 결과를 비교할 수 있다는 뜻입니다.

### 3-1. 데이터 정제 — 왜 분석 전에 자료를 확인하나요?

**정제(Data Cleaning)**는 표에 중복된 날짜나 비어 있는 값, 서로 모순되는 기온 등이 있는지 확인하고 정리하는 과정입니다. 자료가 틀리면 그래프도 잘못될 수 있어서 계산 전에 검사합니다.

- **중복 날짜:** 하루가 두 번 있으면 평균이 잘못될 수 있어 한 번만 사용합니다.
- **누락 날짜:** 하루가 통째로 빠져 있으면 장기 추세를 계산할 때 문제가 생길 수 있어 날짜를 확인합니다.
- **물리적 범위 점검:** 비정상적으로 큰·작은 값은 입력 오류 가능성이 있어 점검용 범위를 적용합니다. 정한 범위 밖의 모든 날이 자연적으로 불가능하다는 뜻은 아닙니다.
- **논리 오류:** 최저기온이 최고기온보다 큰 경우는 서로 맞지 않는 기록이므로 그대로 사용하지 않습니다.
- **보간(Interpolation):** 며칠 빠진 값을 앞뒤 날짜로 추정해 채우는 것입니다. 실제로 측정된 값이 아니므로 이 프로젝트는 *연속 3일 이하의 짧은 결측*에만 허용합니다.
- **통계적 이상치:** 같은 무렵의 평년 기온과 비교해 특별히 다른 날입니다. **표준편차(σ)**는 값이 보통 얼마나 퍼져 있는지 나타내며, ±4σ는 평소의 흔들림보다 크게 벗어난 날을 표시하는 기준입니다. 실제 폭염·한파가 될 수 있으니 무조건 삭제하지 않습니다.

이 프로젝트에서는 아래 기준으로 검사했고, 결과는 코드가 직접 계산했습니다.

| 단계 | 기준 | 결과 |
|---|---|---|
| 중복 날짜 | 같은 날짜가 2번 이상이면 첫 행만 유지 | {log['duplicates_removed']}건 제거 |
| 시간 축 정렬 | 시작~끝 모든 날짜를 만들고 빠진 날짜는 빈 행으로 추가 | {log['dates_added']}일 추가 |
| 물리적 이상치 | 기온이 {PHYSICAL_RANGE[0]:.0f}~{PHYSICAL_RANGE[1]:.0f}℃ 밖이거나 강수가 음수면 결측 처리 | {kv(log['out_of_range_to_nan'])} |
| 논리 오류 | 최저기온 > 최고기온이면 둘 다 결측 처리 | {log.get('tmin_gt_tmax', 0)}건 |
| 결측 보간 | 연속 {MAX_INTERP_GAP}일 이하 공백만 선형 보간, 더 긴 공백은 비워 둠 (긴 공백을 채우면 없는 패턴이 생김) | 보간: {kv(log['interpolated'])} / 남은 결측: {kv(log['remaining_missing'])} |
| 통계적 이상치 | 같은 날짜의 평년값 대비 ±{ANOMALY_Z:.0f}σ를 벗어나면 **표시만 하고 유지** (극한 기상은 분석 대상이므로 삭제하지 않음) | {log['statistical_outliers_flagged']}일 (예: {outlier_dates}) |

## 4. 분석 결과 및 시각화

### 4-1. 일평균 기온과 이동평균 — 추세·계절성·노이즈 한눈에 보기 (Q1)

![일평균 기온과 이동평균]({imgs['daily']})

**이 그래프를 왜 그리나요?** 하루 평균기온은 여름과 겨울의 차이가 커서 25년 동안의 작은 변화를 알아보기 어렵습니다. 그래서 **30일 이동평균**(근처 30일씩 묶어 낸 평균)과 **365일 이동평균**(근처 약 1년씩 묶어 낸 평균)을 같이 그려 장기 변화를 더 쉽게 보게 했습니다. 옅은 선이 원래의 매일 기온이고, 평균선이 단기·장기 흐름입니다.

- 원자료(옅은 선)는 하루하루 날씨 때문에 들쭉날쭉하다(**노이즈**).
- **30일 이동평균**은 노이즈를 걷어내 매년 반복되는 여름–겨울 파동(**계절성**)을 보여 준다.
- **365일 이동평균**은 계절성까지 1년 단위로 상쇄해 **장기 추세**만 남긴다. 범위는 {dec['trend'].min():.1f}~{dec['trend'].max():.1f}℃이다.
- 집계 단위를 일 → 30일 → 365일로 바꾸면 같은 데이터에서 보이는 패턴이 달라진다. 그래서 질문에 맞춰 창 크기를 골랐다.

### 4-2. 연평균 기온과 선형 추세 (Q1)

![연평균 기온 추세]({imgs['annual']})

**이 그래프를 왜 그리나요?** 하루 기온 대신 한 해의 평균을 하나의 점으로 표시하면 계절 변화가 덜 두드러집니다. **선형 추세**는 그 점들에 가장 잘 들어맞는 직선으로, 한 해의 더위가 아니라 여러 해의 변화 방향을 확인합니다. **10년당 변화량**은 그 직선이 10년 동안 얼마나 올라가거나 내려가는지를 뜻합니다. **95% 범위**는 계산된 추세의 불확실성을 나타내는 *통계적 추정 구간*이며, 미래 기온이 그 안에 95% 들어간다는 뜻은 아닙니다.

- 연평균 기온의 선형 추세는 **{per_decade:+.2f}℃/10년**이다. {ci_text}.
- 처음 5년({annual.index[0]}~{annual.index[4]}) 평균은 {first5.mean():.2f}℃, 마지막 5년({annual.index[-5]}~{annual.index[-1]}) 평균은 {last5.mean():.2f}℃로 **{last5.mean() - first5.mean():+.2f}℃** 차이가 난다.
- 가장 더운 해 상위 3개: {', '.join(f'{y}년({v:.2f}℃)' for y, v in warm_years.items())} / 가장 추운 해: {cold_year}년({annual['tmean'].min():.2f}℃)
- y축은 연평균 범위에 맞췄다. 0℃부터 그리면 1℃ 안팎의 변화가 평평하게 보여 추세를 읽을 수 없기 때문이다.

### 4-3. 월별 평균 기온: 과거 vs 최근 (Q2)

![월별 구간 비교]({imgs['monthly']})

**이 그래프를 왜 그리나요?** 25년 동안 연평균만 보면 겨울과 봄 중 어디에서 기온이 더 크게 달라졌는지 알 수 없습니다. **구간별 통계**는 전체 기간 중 두 구간을 정해 같은 달끼리 평균을 비교하는 방법입니다. 예를 들어 두 기간의 3월 평균을 비교해 '3월이 얼마나 달라졌나'를 확인합니다. 이는 변화가 계속 일정하게 일어났다는 증거는 아닙니다.

- 12개월 중 **{n_warmer_months}개월**은 최근 구간({PERIOD_2[0]}~{PERIOD_2[1]})이 과거 구간({PERIOD_1[0]}~{PERIOD_1[1]})보다 따뜻하다.
- 가장 많이 오른 달: **{max_m}월 ({pm.loc[max_m, 'diff']:+.2f}℃)** / 가장 적게 오르거나 내린 달: **{min_m}월 ({pm.loc[min_m, 'diff']:+.2f}℃)**
- 두 구간 사이에 5년 간격(2010~2014)을 둔 것은 경계 부근의 해가 결과를 좌우하지 않게 하기 위해서다.

### 4-4. 연도별 극단 기온일 수 (Q3)

![극단 기온일]({imgs['extremes']})

**이 그래프를 왜 그리나요?** 평균이 1℃ 정도 달라져도 생활에서 체감하는 아주 더운 날이나 밤의 **횟수**는 크게 바뀔 수 있습니다. 따라서 미리 정한 기준을 넘는 날을 해마다 직접 셉니다. 이 분석에서 **폭염일**은 최고기온 33℃ 이상, **열대야 근사일**은 그 날짜의 최저기온 25℃ 이상, **한파일**은 최저기온 -12℃ 이하인 날로 계산했습니다. *기상청 공식 열대야 정의는 야간 시간대를 기준으로 하므로, 여기의 일 최저기온 기준과 같지 않습니다.* 아래의 '연평균'은 각 기간에 해마다 해당하는 날이 평균 몇 일인지 뜻합니다.

| 지표 | {PERIOD_1[0]}~{PERIOD_1[1]} 연평균 | {PERIOD_2[0]}~{PERIOD_2[1]} 연평균 | 변화 | 최다 연도 |
|---|---|---|---|---|
| 폭염일 (최고 ≥{HEAT_DAY:.0f}℃) | {heat1:.1f}일 | {heat2:.1f}일 | {ratio_text(heat1, heat2)} | {heat_top}년 ({int(annual.loc[heat_top, 'heat_days'])}일) |
| 열대야 (최저 ≥{TROPICAL_NIGHT:.0f}℃) | {trop1:.1f}일 | {trop2:.1f}일 | {ratio_text(trop1, trop2)} | {trop_top}년 ({int(annual.loc[trop_top, 'tropical_nights'])}일) |
| 한파일 (최저 ≤{COLD_DAY:.0f}℃) | {cold1:.1f}일 | {cold2:.1f}일 | {ratio_text(cold1, cold2)} | {annual['cold_days'].idxmax()}년 ({int(annual['cold_days'].max())}일) |

### 4-5. 하루 사이 기온 변동성 (Q4)

![월별 변동성]({imgs['volatility']})

**이 그래프를 왜 그리나요?** 같은 10℃라도 날마다 9℃·11℃를 오가는 경우와 2℃·18℃를 오가는 경우는 매우 다릅니다. 그래서 **전일 대비 변화량**(오늘의 평균기온에서 어제의 평균기온을 뺀 수치)을 먼저 구한 뒤, 그 차이들이 얼마나 퍼져 있는지 **표준편차**로 비교했습니다. 표준편차가 큰 달은 하루 사이 기온 변화 폭이 들쭉날쭉한 경향이 강한 달입니다.

- 기온은 0℃ 근처에서 부호가 바뀌어 **%변화율이 의미가 없다**(예: 0.5℃ → 1℃는 +100%). 그래서 **전일 대비 변화량(℃)**을 변화 지표로 썼다.
- 변동성(변화량 표준편차)이 가장 큰 달은 **{vmax_m}월({vol.loc[vmax_m, 'std']:.2f}℃)**, 가장 작은 달은 **{vmin_m}월({vol.loc[vmin_m, 'std']:.2f}℃)**이다.
- 하루 만에 가장 크게 떨어진 기록은 {vol['max_drop'].min():.1f}℃({int(vol['max_drop'].idxmin())}월), 가장 크게 오른 기록은 {vol['max_rise'].max():+.1f}℃({int(vol['max_rise'].idxmax())}월)이다.

### 4-6. (보너스) 시계열 분해 — 추세 / 계절성 / 잔차

![시계열 분해]({imgs['decomposition']})

**보너스 과제를 왜 하나요?** 같은 서울 기온이라도 '25년 동안의 변화', '여름·겨울마다 반복되는 변화', '특정 날의 예상 밖 변화'가 한 선에 겹쳐 있습니다. **시계열 분해**란 그 선을 아래 세 종류로 나누어 따로 보는 방법입니다.

1. **추세(Trend):** 365일 중심 이동평균으로 추정한 장기적인 높낮이
2. **계절성(Seasonality):** 같은 달·날짜에 반복해서 나타나는 일반적인 더위·추위 패턴
3. **잔차(Residual):** 실제 기온에서 추세와 계절성을 뺀 나머지. 오차만을 뜻하는 것이 아니라 그날의 불규칙한 날씨 등도 포함합니다.

**가법 분해**는 세 요소를 *더하면 원래 값이 되도록* 나누는 방식입니다. **계절성 진폭**은 계절성의 가장 높은 값과 가장 낮은 값의 차이, **잔차 표준편차**는 나머지가 보통 얼마나 크게 흔들리는지를 뜻합니다. 이 보너스는 추세와 계절성 구분을 설명하는 것이지, 기온이 변한 원인을 증명하는 방법은 아닙니다.

- 고전적 가법 분해를 썼다. 관측값 = 추세(365일 중심 이동평균) + 계절성(날짜별 평균 패턴) + 잔차.
- **계절성 진폭(최고−최저)은 약 {seasonal_amp:.1f}℃**로, 25년 동안 추세가 움직인 폭({trend_range:.1f}℃)보다 훨씬 크다. 잔차(노이즈)의 표준편차는 {resid_std:.1f}℃다.
- **해석:** 서울 기온의 하루 값은 대부분 '몇 월 며칠인가'(계절성)와 '그날 날씨'(노이즈)로 결정된다. 장기 추세는 그 위에 얹힌 작은 이동이라, 일별 그래프만 보면 추세가 잘 보이지 않는다. 4-2처럼 연 단위로 집계해야 드러난다.

### 4-7. (보너스) 베이스라인 예측 — 마지막 {FORECAST_HORIZON}일 맞혀 보기

![베이스라인 예측]({imgs['forecast']})

**이 보너스를 왜 하나요?** 먼저 예전 날짜만 보고 미래 며칠을 예측한 뒤, 실제 기온과 비교하면 단순한 계절 정보만으로도 어느 정도 맞힐 수 있는지 알 수 있습니다. **베이스라인(Baseline)**은 복잡한 AI 모델과 비교할 때 쓰는 간단한 기준 예측법입니다. **MAE(평균절대오차)**는 예측과 실제 기온의 차이를 크기만 취해 평균 낸 값으로, 이번 실험에서는 숫자가 작을수록 예측이 더 잘 맞았다는 뜻입니다.

{fc['train_end']}까지만 보고 {fc['test_start']}~{fc['test_end']}를 예측해 평균절대오차(MAE)를 비교했다.

| 방법 | MAE (℃) |
|---|---|
{mae_rows}

- **평년값:** 과거 여러 해 같은 날짜의 평균 기온을 예상값으로 쓰는 방법
- **평년값 + 최근 편차:** 평년값에 최근 며칠의 따뜻함·추움 정도를 보정하는 방법
- **마지막 값 유지:** 다음 날도 마지막으로 본 날과 같다고 가정하는 방법
- **작년 같은 날:** 작년 같은 날짜를 참고하는 방법
- 이 한 번의 실험에서 MAE가 가장 작았던 방법: **{model_ko[best_model]}**
- **가정:** '평년값' 계열은 계절 패턴이 해마다 같다고 가정한다. '마지막 값 유지'는 날씨가 그대로 이어진다고 가정한다.
- **한계:** 예측 구간이 {FORECAST_HORIZON}일 한 번뿐이라 운이 크게 작용한다. 연말처럼 기온이 빠르게 떨어지는 시기에는 '마지막 값 유지'가 구조적으로 불리하다. 정확도 비교보다는 "계절성을 아는 것만으로 오차가 얼마나 줄어드는가"를 보는 실험이다.

## 5. 인사이트 — 숫자에서 어떤 의미를 읽을 수 있나요?

**인사이트(Insight)**는 그래프에 숫자를 표시한 다음, *그 숫자가 무엇을 의미하고 앞으로 무엇을 더 확인해야 하는지* 정리한 내용입니다. **관찰(Fact)**은 계산값이므로 데이터로 확인할 수 있지만, **원인 가설(Why)**은 아직 가능성을 설명하는 문장입니다. **행동(Action)**은 추가로 검증할 일입니다. 아래 가설은 연구 결과로 확정된 원인이 아닙니다.

### 인사이트 1. {insight1_title} (Q1)

- **관찰(Fact):** 연평균 기온 선형 추세 {per_decade:+.2f}℃/10년, 처음 5년 대비 마지막 5년 {last5.mean() - first5.mean():+.2f}℃. 가장 더운 해 3개는 {', '.join(str(y) for y in warm_years.index)}년이다.
- **원인 가설(Why):** ① 지구 평균기온 상승이라는 큰 흐름 ② 도시화에 따른 열섬 효과. 단, ERA5는 약 25km 격자 평균이라 도심 열섬은 일부만 반영된다. 실제 관측소 값은 이보다 클 수 있다. ③ 특정 해가 유독 높다면 엘니뇨 같은 해양 변동의 영향일 수 있다.
- **행동(Action):** 기상청 서울 관측소(ASOS 108) 자료로 같은 분석을 반복해 격자 자료와 비교한다. 기간을 1973년 이후로 늘려 추세가 유지되는지 본다.

### 인사이트 2. 기온 변화는 계절마다 고르지 않으며, {max_m}월({season_max})에 가장 크게 올랐다 (Q2)

- **관찰(Fact):** 두 구간 월평균 차이는 {max_m}월이 {pm.loc[max_m, 'diff']:+.2f}℃로 가장 크고, {min_m}월이 {pm.loc[min_m, 'diff']:+.2f}℃로 가장 작다. 12개월 중 {n_warmer_months}개월이 따뜻해졌다.
- **원인 가설(Why):** {SEASON_HYPOTHESIS[season_max]}. 반례 점검: 한 달 평균은 10년치 표본(10개)뿐이라 특정 해 하나가 결과를 크게 바꿀 수 있다.
- **행동(Action):** 월별로 따로 연도 추세를 구해 한두 해의 영향인지, 꾸준한 변화인지 확인한다. 냉난방 수요나 계절 상품 판매 계획처럼 '어느 달이 변했는가'가 중요한 의사결정에는 연평균보다 월별 지표를 쓴다.

### 인사이트 3. {insight3_title} (Q3)

- **관찰(Fact):** 연평균 폭염일은 {heat1:.1f}일 → {heat2:.1f}일({ratio_text(heat1, heat2)}), 열대야는 {trop1:.1f}일 → {trop2:.1f}일({ratio_text(trop1, trop2)}), 한파일은 {cold1:.1f}일 → {cold2:.1f}일이다.
- **원인 가설(Why):** {insight3_why}
- **주의(반례):** ERA5 격자값은 주변을 평균한 값이라 관측소보다 최고기온은 낮게, 최저기온은 높게 나오는 경향이 있다. 따라서 **일수의 절대값은 기상청 발표와 다를 수 있다.** 같은 자료 안에서의 증감 비교로만 해석해야 한다.
- **행동(Action):** 폭염 대응·전력 수요 계획에는 평균보다 기준 초과 일수 지표를 쓴다. 관측소 자료로 일수를 다시 세어 비교한다.

### 인사이트 4. 하루 사이 기온 변화가 가장 들쭉날쭉한 달은 {vmax_m}월이다 (Q4)

- **관찰(Fact):** 전일 대비 변화량의 표준편차는 {vmax_m}월 {vol.loc[vmax_m, 'std']:.2f}℃로 가장 크고, {vmin_m}월 {vol.loc[vmin_m, 'std']:.2f}℃로 가장 작다. 최대 하루 하강폭은 {vol['max_drop'].min():.1f}℃다.
- **원인 가설(Why):** {VOLATILITY_HYPOTHESIS[SEASON_OF_MONTH[vmax_m]]}. 반대로 {vmin_m}월은 {"습한 공기가 오래 머물러 기온이 안정적인" if SEASON_OF_MONTH[vmin_m] == "여름" else "기단이 비교적 안정된"} 시기일 수 있다.
- **행동(Action):** {vmax_m}월에는 '어제와 비슷할 것'이라는 단순 예측이 잘 맞지 않는다. 일정·재고·에너지 계획에서 여유분을 더 둔다. 4-7의 예측 실험도 계절에 따라 결과가 달라질 수 있어, 여러 시점에서 반복해야 한다.

## 6. 결론 및 한계점

**결론**은 분석 질문에 대해 데이터에서 확인한 내용을 짧게 정리한 것입니다. **한계점**은 '이번 자료만으로는 확실하게 말할 수 없는 부분'입니다. 예를 들어 기온이 올라갔다는 관찰이 곧바로 특정 원인이 기온을 올렸다는 증명은 아닙니다.

### 결론

- **Q1:** {q1_conclusion}
- **Q2:** 변화는 계절마다 고르지 않았고, {max_m}월에 가장 컸다.
- **Q3:** 폭염일·열대야는 평균 변화보다 상대적으로 크게 변했다. 극단 지표를 따로 봐야 하는 이유다.
- **Q4:** 하루 사이 변동은 {vmax_m}월에 가장 크고 {vmin_m}월에 가장 작다. 이 시기 차이가 예측 난이도에 영향을 준다.

### 한계점

1. **격자 자료의 한계:** ERA5는 약 25km 격자 평균값이라 도심 열섬과 극한값이 약하게 나타난다. 관측소 자료와 절대값이 다를 수 있다.
2. **기간의 한계:** 25년은 기후 추세를 말하기에 짧다. 엘니뇨 같은 수년 주기 변동이 추세에 섞여 있을 수 있다.
3. **상관 ≠ 인과:** 이 분석은 '변했다'는 것을 보여 줄 뿐, 원인(온난화·도시화·해양 변동)을 분리하지 못했다. Why 항목은 모두 가설이다.
4. **단순 기법:** 선형 추세, 고전적 분해, 베이스라인 예측만 썼다. 자기상관을 고려한 검정이나 계절 모형(SARIMA 등)은 적용하지 않았다.
5. **극단 기준:** 폭염·열대야 기준은 기상청 기준을 일 단위 값으로 근사한 것이다. 열대야의 공식 정의(18시~익일 09시 최저기온)와 정확히 같지 않다.

## 7. AI 사용 로그 — AI를 어디에 사용했고 어떻게 확인했나요?

**AI 사용 로그**는 AI가 도와준 작업, AI를 쓴 이유, AI의 설명·코드가 맞는지 사람이 직접 확인한 방법을 기록하는 표입니다. 코드를 만들었다고 자동으로 옳다고 판단하지 않고, 원본 자료와 계산 결과를 비교해야 합니다.

> 직접 확인한 항목에 `[x]` 표시하고, 실제로 한 내용으로 고쳐 쓰세요.

| 사용 작업 | 사용 이유 | 검증 방법 |
|---|---|---|
| 주제·데이터 소스 선정 (Open-Meteo) | 키 없이 받을 수 있는 시계열 대안 탐색 | [ ] Open-Meteo 문서에서 변수명·라이선스 직접 확인 |
| 수집·정제·분석 코드 작성 (`fetch_data.py`, `analysis.py`) | 시간 절감 | [ ] 임의의 날짜 3개를 기상자료개방포털 서울(108) 값과 비교 (격자 차이 감안) / [ ] 연평균 1개를 엑셀로 따로 계산해 일치 확인 |
| 이동평균 창 크기, %변화율 대신 ℃ 변화량 사용 결정 | 대안 비교 | [ ] 창 크기를 7·90일로 바꿔 그래프 비교 |
| 인사이트 원인 가설 초안 | 해석 대안 탐색 | [ ] 각 가설을 뒷받침/반박하는 그래프 수치를 직접 확인하고 동의하지 않는 문장 수정 |
| 리포트 문장 다듬기 | 가독성 | [ ] 모든 수치가 그래프와 일치하는지 확인 |

## 8. 재현 방법 — 다른 사람도 같은 결과를 얻으려면?

**재현**은 같은 CSV와 코드로 다시 계산해 비슷한 그래프와 수치가 나오는지 확인하는 것입니다. `requirements.txt`에는 필요한 Python 라이브러리가 적혀 있습니다. 저장된 CSV만으로 재계산할 수 있으므로, 이미 자료가 있다면 `fetch_data.py`를 반드시 다시 실행할 필요는 없습니다.

```bash
python -m pip install -r requirements.txt
python fetch_data.py        # 선택: 인터넷에서 자료 다시 받기 → data/ (기존 CSV 덮어쓸 수 있음)
python analysis.py          # CSV 재분석 → images/, results/, REPORT.md
python -m streamlit run dashboard.py  # 선택: 웹 대시보드 실행
```

**중요:** `analysis.py`를 다시 실행하면 지금 읽는 `REPORT.md`가 코드에 포함된 설명과 최신 계산값으로 **자동 재생성**됩니다. 보고서를 직접 편집했다면 실행 전에 백업해 주세요. **자료 수집을 다시 했다면** `data/metadata.json`에서 사용 모델이 `era5`인지 확인하세요.
"""


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main() -> None:
    setup_font()
    print("📊 서울 기온 시계열 분석\n")

    raw = load_raw()
    info = inspect(raw)
    print(f"[1/6] 데이터 로드: {info['rows']:,}행, {info['start']} ~ {info['end']}")
    print(f"      컬럼: {info['columns']}")
    print(f"      결측: {info['missing']}, 중복 날짜: {info['duplicate_dates']}, 빠진 날짜: {info['missing_dates']}")
    if info["rows"] < 100:
        raise SystemExit("❌ 데이터 포인트가 100개 미만입니다.")

    df, log = clean(raw)
    print(f"[2/6] 정제 완료: 보간 {log['interpolated']}, 남은 결측 {log['remaining_missing']}, "
          f"통계적 이상치 표시 {log['statistical_outliers_flagged']}일")

    df = add_features(df)
    annual = annual_stats(df)
    trend = linear_trend(np.array(annual.index, dtype=float), annual["tmean"].to_numpy())
    pm = period_monthly(df)
    vol = monthly_volatility(df)
    print(f"[3/6] 시계열 분석: 연평균 추세 {trend['slope'] * 10:+.2f}℃/10년, "
          f"최대 상승 월 {int(pm['diff'].idxmax())}월, 변동성 최대 월 {int(vol['std'].idxmax())}월")

    dec = decompose(df)
    preds, fc = baseline_forecast(df)
    print(f"[4/6] 보너스: 분해 완료, 예측 MAE {', '.join(f'{k}={v:.2f}' for k, v in fc['mae'].items())}")

    imgs = {
        "daily": plot_daily(df),
        "annual": plot_annual_trend(annual, trend),
        "monthly": plot_monthly_compare(pm),
        "extremes": plot_extremes(annual),
        "volatility": plot_volatility(df, vol),
        "decomposition": plot_decomposition(dec),
        "forecast": plot_forecast(df, preds, fc),
    }
    print(f"[5/6] 그래프 {len(imgs)}개 저장: {IMG_DIR}")

    meta_path = DATA_PATH.parent / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    RESULT_DIR.mkdir(exist_ok=True)
    summary = {
        "data_info": info, "cleaning_log": log,
        "annual_trend_per_decade": trend["slope"] * 10, "annual_trend": trend,
        "annual": annual.reset_index().to_dict(orient="records"),
        "monthly_period_comparison": pm.reset_index().rename(columns={"index": "month"}).to_dict(orient="records"),
        "monthly_volatility": vol.reset_index().rename(columns={"index": "month"}).to_dict(orient="records"),
        "forecast": fc,
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")
    annual.to_csv(RESULT_DIR / "annual_stats.csv", encoding="utf-8-sig")

    REPORT_PATH.write_text(build_report(info, log, annual, trend, pm, vol, dec, fc, imgs, meta), encoding="utf-8")
    print(f"[6/6] 리포트 생성: {REPORT_PATH}")
    print("\n✅ 완료! REPORT.md의 '원인 가설'과 'AI 사용 로그'를 직접 검토·수정하세요.")


if __name__ == "__main__":
    main()
