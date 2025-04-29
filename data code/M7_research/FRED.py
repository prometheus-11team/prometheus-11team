from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# API 키 로딩
fred_api_key = os.getenv('FRED_API_KEY')

# FRED 객체 생성
fred = Fred(api_key=fred_api_key)

# 주요 거시경제 지표 코드 (M7에 맞춘 추천 리스트)
indicators = {
    'FEDFUNDS': 'Federal Funds Rate',
    'GS10': '10Y Treasury Yield',
    'CPIAUCSL': 'CPI',
    'CPILFESL': 'Core CPI',
    'PCEPI': 'PCE Price Index',
    'RSAFS': 'Retail Sales',
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Non-Farm Payrolls',
    'NAPM': 'ISM Manufacturing PMI',  
    'M2SL': 'M2 Money Stock'
}

# 기존 파일 경로
file_path = '../store data/macro_indicators_2020_2024.csv'

# 기존 데이터 불러오기 (만약 존재하면)
if os.path.exists(file_path):
    print(f"기존 파일 {file_path} 불러오는 중...")
    df_existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
else:
    print(f"기존 파일 {file_path} 없음. 새로 생성합니다.")
    df_existing = pd.DataFrame()

# 새로운 데이터 가져오기
data_new = {}

for code, name in indicators.items():
    try:
        print(f"\n[{name}] 데이터를 가져오는 중...")
        series = fred.get_series(code, observation_start='2020-01-01', observation_end='2025-04-28')
        data_new[name] = series
        print(f"[{name}] 데이터 성공")
    except Exception as e:
        print(f"[{name}] 실패 -> 코드: {code}, 에러: {e}")

# 새로 가져온 데이터를 데이터프레임으로 변환
df_new = pd.DataFrame(data_new)

# 기존 데이터와 합치기
df_combined = pd.concat([df_existing, df_new])

# 중복 인덱스(날짜) 제거 (최근 데이터로 덮어쓰기)
df_combined = df_combined[~df_combined.index.duplicated(keep='last')]

# 저장
df_combined.to_csv(file_path)

print("\n[완료] macro_indicators_2020_2024.csv 파일 업데이트 및 저장 완료!")
print(df_combined.tail())
