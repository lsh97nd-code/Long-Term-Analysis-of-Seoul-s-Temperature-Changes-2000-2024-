"""
(보너스) 서울 기온 탐색 대시보드

실행: streamlit run dashboard.py
     → 브라우저에서 http://localhost:8501 이 열린다.

사이드바에서 기간·지표·이동평균 창·폭염 기준·월을 바꿔 가며 탐색할 수 있다.
분석 로직은 analysis.py의 함수를 그대로 재사용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import analysis as an

st.set_page_config(page_title="서울 기온 트렌드 대시보드", page_icon="🌡️", layout="wide")

METRICS = {"일평균 기온 (tmean)": "tmean", "일 최고기온 (tmax)": "tmax", "일 최저기온 (tmin)": "tmin"}


@st.cache_data
def load() -> pd.DataFrame:
    raw = an.load_raw()
    df, _ = an.clean(raw)
    return an.add_features(df)


try:
    data = load()
except FileNotFoundError as e:
    st.error(f"{e}\n\n터미널에서 `python fetch_data.py`를 먼저 실행하세요.")
    st.stop()

# ── 사이드바: 조건 ─────────────────────────────
st.sidebar.header("🔧 조건 설정")
y_min, y_max = int(data["year"].min()), int(data["year"].max())
years = st.sidebar.slider("기간 (연도)", y_min, y_max, (y_min, y_max))
metric_label = st.sidebar.selectbox("지표", list(METRICS))
metric = METRICS[metric_label]
window = st.sidebar.slider("이동평균 창 (일)", 7, 365, 30, step=1,
                           help="짧으면 날씨 변동, 30일 전후면 계절성, 365일이면 장기 추세가 보입니다.")
heat_th = st.sidebar.number_input("폭염 기준 (최고기온 ℃ 이상)", value=an.HEAT_DAY, step=0.5)
trop_th = st.sidebar.number_input("열대야 기준 (최저기온 ℃ 이상)", value=an.TROPICAL_NIGHT, step=0.5)
months = st.sidebar.multiselect("월 선택 (비우면 전체)", list(range(1, 13)), format_func=lambda m: f"{m}월")

sel = data[(data["year"] >= years[0]) & (data["year"] <= years[1])].copy()
if months:
    sel = sel[sel["month"].isin(months)]

st.title("🌡️ 서울 기온 트렌드 대시보드")
st.caption("데이터: Open-Meteo Historical Weather API (ERA5 재분석, CC BY 4.0) · 서울시청 부근 격자")

if sel.empty or sel[metric].notna().sum() < 2:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다. 기간이나 월을 넓혀 보세요.")
    st.stop()

# ── 핵심 지표 ─────────────────────────────────
annual = sel.groupby("year").agg(value=(metric, "mean"),
                                 heat=("tmax", lambda s: int((s >= heat_th).sum())),
                                 trop=("tmin", lambda s: int((s >= trop_th).sum())))
c1, c2, c3, c4 = st.columns(4)
c1.metric("평균", f"{sel[metric].mean():.1f}℃")
c2.metric("최고 / 최저", f"{sel[metric].max():.1f} / {sel[metric].min():.1f}℃")
if len(annual) >= 3:
    tr = an.linear_trend(np.array(annual.index, dtype=float), annual["value"].to_numpy())
    c3.metric("연도별 추세", f"{tr['slope'] * 10:+.2f}℃/10년", help="선택 기간·월의 연평균에 직선을 맞춘 기울기")
else:
    c3.metric("연도별 추세", "3년 이상 선택 필요")
c4.metric("연평균 폭염일 / 열대야", f"{annual['heat'].mean():.1f} / {annual['trop'].mean():.1f}일")

# ── 1. 시계열 + 이동평균 ───────────────────────
st.subheader(f"① {metric_label} 와 {window}일 이동평균")
line = pd.DataFrame({"원자료": sel[metric],
                     f"{window}일 이동평균": data[metric].rolling(window, min_periods=int(window * 0.9)).mean()
                     .reindex(sel.index)})
st.line_chart(line, height=320)
st.caption("이동평균은 선택 구간 밖의 앞쪽 데이터까지 포함해 계산하므로, 구간 시작 부분도 끊기지 않습니다.")

# ── 2. 연도별 집계 / 3. 월별 비교 ───────────────
left, right = st.columns(2)
with left:
    st.subheader("② 연도별 평균과 극단 일수")
    st.bar_chart(annual[["heat", "trop"]].rename(columns={"heat": f"폭염일(≥{heat_th}℃)",
                                                           "trop": f"열대야(≥{trop_th}℃)"}), height=280)
    st.line_chart(annual["value"].rename(f"연평균 {metric}"), height=200)
with right:
    st.subheader("③ 월별 평균: 선택 기간 vs 전체 기간")
    monthly = pd.DataFrame({
        "선택 기간": sel.groupby("month")[metric].mean(),
        "전체 기간": data.groupby("month")[metric].mean(),
    }).reindex(range(1, 13))
    monthly.index = [f"{m:02d}월" for m in monthly.index]
    st.bar_chart(monthly.dropna(how="all"), height=280, stack=False)
    vol = sel.groupby("month")["change"].std().reindex(range(1, 13))
    vol.index = [f"{m:02d}월" for m in vol.index]
    st.caption("월별 변동성 (전일 대비 변화량 표준편차, ℃)")
    st.bar_chart(vol.dropna().rename("변동성"), height=200)

# ── 탐색 시나리오 안내 ─────────────────────────
with st.expander("💡 이렇게 탐색해 보세요"):
    st.markdown(
        "1. **기간을 2000–2009 → 2015–2024로 바꿔** 연평균 폭염일과 열대야 수가 어떻게 달라지는지 비교합니다.\n"
        "2. **월을 7·8월만 선택**하고 지표를 최저기온으로 바꿔 여름밤 기온 추세를 봅니다.\n"
        "3. **이동평균 창을 7 → 30 → 365일로 늘려** 노이즈 → 계절성 → 추세 순서로 보이는 것이 달라지는 것을 확인합니다.\n"
        "4. **폭염 기준을 31℃, 35℃로 바꿔** 기준에 따라 '늘었다'는 결론이 유지되는지 확인합니다."
    )

with st.expander("📄 선택 데이터 보기 / 내려받기"):
    view = sel[["tmax", "tmin", "tmean", "precip", "change"]].round(2)
    st.dataframe(view, height=300)
    st.download_button("CSV 내려받기", view.to_csv().encode("utf-8-sig"),
                       file_name=f"seoul_{years[0]}_{years[1]}.csv", mime="text/csv")
