# 데이터 설명 및 수집 방법

## 출처

| 항목 | 내용 |
|---|---|
| 제공처 | Open-Meteo Historical Weather API — https://open-meteo.com/en/docs/historical-weather-api |
| 원천 자료 | ECMWF ERA5 재분석 자료 (약 25km 격자, 1940년~) |
| 지점 | 서울시청 부근 위도 37.5665, 경도 126.9780 |
| 기간 | 2000-01-01 ~ 2024-12-31 (일 단위) |
| 시간대 | Asia/Seoul (하루 = 한국 시간 0~24시) |

## 수집 방법

```bash
python fetch_data.py
```

요청 예시 (브라우저에 붙여 넣어도 결과가 나옵니다):

```
https://archive-api.open-meteo.com/v1/archive?latitude=37.5665&longitude=126.978&start_date=2000-01-01&end_date=2024-12-31&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum&timezone=Asia%2FSeoul&models=era5
```

- `models=era5`: 전 기간을 같은 모델로 받아 연도 간 비교가 왜곡되지 않게 합니다. 기본값을 쓰면 2017년 이후 고해상도 모델이 섞입니다.
- 실제 사용한 URL은 `metadata.json`의 `request.url`에 기록됩니다.

## 컬럼

| 컬럼 | 설명 | 단위 |
|---|---|---|
| date | 날짜 | YYYY-MM-DD |
| tmax | 일 최고기온 (지상 2m) | ℃ |
| tmin | 일 최저기온 (지상 2m) | ℃ |
| tmean | 일 평균기온 (지상 2m) | ℃ |
| precip | 일 강수량 합계 | mm |

## 라이선스 · 주의 사항

- **CC BY 4.0** — 재사용할 때는 "Weather data by Open-Meteo.com"과 "Generated using Copernicus Climate Change Service information"을 표기해야 합니다.
- Open-Meteo 무료 API는 **비상업적 용도**에 한해 쓸 수 있습니다.
- ERA5는 관측소 값이 아니라 **격자 평균 재분석 값**입니다. 기상청 서울 관측소(ASOS 108) 값과 다를 수 있고, 특히 극한 최고·최저기온이 완만하게 나타납니다.
