"""
서울 일별 기온·강수 데이터 수집 스크립트

데이터 출처: Open-Meteo Historical Weather API (https://open-meteo.com/en/docs/historical-weather-api)
  - API 키 없이 사용 가능 (비상업적 용도)
  - 원천 데이터: ECMWF ERA5 재분석 자료 (약 25km 격자)
  - 라이선스: CC BY 4.0, 출처 표기 필요
    "Weather data by Open-Meteo.com", "Generated using Copernicus Climate Change Service information"

실행: python fetch_data.py
결과: data/seoul_weather_2000_2024.csv

외부 패키지 없이 표준 라이브러리만 사용한다.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ── 수집 조건 ──────────────────────────────────
CITY = "Seoul"
LATITUDE = 37.5665      # 서울시청
LONGITUDE = 126.9780
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"
TIMEZONE = "Asia/Seoul"
# 연도마다 같은 모델을 써야 장기 추세가 왜곡되지 않는다.
# (기본값 best_match는 2017년 이후 고해상도 모델이 섞임)
MODEL = "era5"

DAILY_VARS = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum"]
COLUMN_NAMES = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "temperature_2m_mean": "tmean",
    "precipitation_sum": "precip",
}

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "data" / f"seoul_weather_{START_DATE[:4]}_{END_DATE[:4]}.csv"
RAW_JSON_PATH = ROOT / "data" / "open_meteo_raw.json"


def build_url(daily_vars: list[str], model: str | None) -> str:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(daily_vars),
        "timezone": TIMEZONE,
    }
    if model:
        params["models"] = model
    return f"{BASE_URL}?{urllib.parse.urlencode(params)}"


def request_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "seoul-temperature-analysis/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch() -> dict:
    """API를 호출한다. 일부 옵션이 거부되면 옵션을 줄여 다시 시도한다."""
    attempts = [
        (DAILY_VARS, MODEL),
        (DAILY_VARS, None),                                   # 모델 지정이 거부된 경우
        ([v for v in DAILY_VARS if v != "temperature_2m_mean"], None),  # mean 변수가 없는 경우
    ]
    last_error = None
    for daily_vars, model in attempts:
        url = build_url(daily_vars, model)
        print(f"요청: {url}")
        try:
            data = request_json(url)
            data["_request"] = {"url": url, "model": model or "best_match(default)"}
            return data
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:200]
            print(f"  ⚠ HTTP {e.code}: {body}")
            last_error = e
            if e.code != 400:      # 400(잘못된 파라미터)일 때만 옵션을 바꿔 재시도
                break
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  ⚠ 네트워크 오류: {e}")
            last_error = e
            break
    raise RuntimeError(last_error)


def load_manual_json() -> dict | None:
    """자동 다운로드가 막힌 환경을 위해, 브라우저로 저장한 JSON 파일을 읽는다."""
    if RAW_JSON_PATH.exists():
        print(f"수동 저장된 파일을 사용합니다: {RAW_JSON_PATH}")
        return json.loads(RAW_JSON_PATH.read_text(encoding="utf-8"))
    return None


def save_csv(data: dict) -> int:
    daily = data["daily"]
    dates = daily["time"]
    cols = [v for v in DAILY_VARS if v in daily]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date"] + [COLUMN_NAMES[c] for c in cols])
        for i, d in enumerate(dates):
            w.writerow([d] + ["" if daily[c][i] is None else daily[c][i] for c in cols])
    return len(dates)


def main() -> None:
    print(f"📥 {CITY} 일별 기온 데이터 수집 ({START_DATE} ~ {END_DATE})\n")
    try:
        data = fetch()
    except RuntimeError:
        data = load_manual_json()
        if data is None:
            print(
                "\n❌ 데이터를 자동으로 받지 못했습니다.\n"
                "   아래 주소를 브라우저에 붙여 넣고, 나온 내용을 다음 파일로 저장한 뒤 다시 실행하세요.\n"
                f"   저장 위치: {RAW_JSON_PATH}\n"
                f"   주소: {build_url(DAILY_VARS, MODEL)}"
            )
            sys.exit(1)

    if "daily" not in data:
        print(f"❌ 응답에 daily 데이터가 없습니다: {str(data)[:200]}")
        sys.exit(1)

    n = save_csv(data)
    meta = {
        "source": "Open-Meteo Historical Weather API (ERA5 reanalysis)",
        "request": data.get("_request", {"url": "manual download"}),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "elevation_m": data.get("elevation"),
        "units": data.get("daily_units"),
        "rows": n,
        "license": "CC BY 4.0 - Weather data by Open-Meteo.com / Copernicus Climate Change Service (ERA5)",
    }
    (OUT_PATH.parent / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ {n}일치 저장 완료: {OUT_PATH}")
    print(f"   메타데이터: {OUT_PATH.parent / 'metadata.json'}")


if __name__ == "__main__":
    main()
