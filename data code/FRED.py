from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# API 키 로딩 확인
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

data = {}

# 디버깅용 루프
for code, name in indicators.items():
    try:
        print(f"\n⏳ [{name}] 데이터를 가져오는 중...")
        series = fred.get_series(code, observation_start='2020-01-01', observation_end='2024-12-31')
        data[name] = series
        print(f"✅ [{name}] 데이터 성공")
    except Exception as e:
        print(f"❌ [{name}] 실패 -> 코드: {code}, 에러: {e}")

# 데이터 합치기 (성공한 것만)
df_macro = pd.DataFrame(data)

# CSV로 저장 (store data 폴더에)
df_macro.to_csv('../store data/macro_indicators_2020_2024.csv')

print("\n[완료] macro_indicators_2020_2024.csv 파일 저장됨!")
print(df_macro.tail())
