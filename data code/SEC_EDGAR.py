import requests
import pandas as pd
import time
# import json
from datetime import datetime

# M7 기업 CIK 목록
M7_CIK = {
    "Apple": "0000320193",
    "Microsoft": "0000789019",
    "Alphabet": "0001652044",
    "Amazon": "0001018724",
    "Meta": "0001326801",
    "Nvidia": "0001045810",
    "Tesla": "0001318605"
}

# 필터링할 회계 연도 범위 (2020-2024)
FISCAL_YEARS = list(range(2020, 2025))

# SEC EDGAR API 기본 URL
BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"

# User-Agent 헤더 설정 (필수)
HEADERS = {
    "User-Agent": "Your Name your@email.com",  # 실제 이름과 이메일로 교체해야 함
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

# 주요 재무 지표 매핑 (US-GAAP 태그명)
FINANCIAL_METRICS = {
    "OperatingIncomeLoss": "Operating Income",
    "NetIncomeLoss": "Net Income",
    "Revenues": "Total Revenue",
    "Revenue": "Total Revenue",  # 대체 태그
    "GrossProfit": "Gross Profit",
    "EarningsPerShareBasic": "EPS Basic",
    "EarningsPerShareDiluted": "EPS Diluted",
    "Assets": "Total Assets",
    "Liabilities": "Total Liabilities",
    "StockholdersEquity": "Shareholders Equity"
}

# 재무 데이터 가져오기 함수
def get_company_financial_data(company_name, cik):
    url = BASE_URL.format(cik.zfill(10))  # CIK를 10자리로 맞추기
    
    print(f"{company_name}({cik})의 재무 데이터를 가져오는 중...")
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        # API 속도 제한을 피하기 위한 대기
        time.sleep(0.1)
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"{company_name}의 데이터 가져오기 실패: {e}")
        return None

# 회사 재무 데이터에서 필요한 지표 추출
def extract_financial_metrics(company_name, data):
    if not data:
        return []
    
    results = []
    company_facts = data.get('facts', {})
    us_gaap = company_facts.get('us-gaap', {})
    
    # 각 회계연도별 데이터를 임시 저장할 딕셔너리
    year_data = {year: {"Company": company_name} for year in FISCAL_YEARS}
    
    # US-GAAP 태그에서 필요한 지표 추출
    for metric_tag, entries in us_gaap.items():
        # 현재 지표가 관심 있는 지표인지 확인
        display_name = FINANCIAL_METRICS.get(metric_tag)
        if not display_name:
            continue
        
        # 단위 데이터 처리 (주로 USD 또는 USD/share)
        for unit_type, unit_data in entries.get('units', {}).items():
            for entry in unit_data:
                # 10-K(연간 보고서) 데이터만 추출하고 회계연도가 대상 범위인지 확인
                if entry.get('form') == '10-K' and entry.get('fy') in FISCAL_YEARS:
                    fiscal_year = entry.get('fy')
                    value = entry.get('val')
                    end_date = entry.get('end')
                    
                    # 중복 처리: 이미 값이 있으면 가장 최근 filing 데이터 사용
                    if display_name not in year_data[fiscal_year] or end_date > year_data[fiscal_year].get('end_date', ''):
                        year_data[fiscal_year][display_name] = value
                        year_data[fiscal_year]['end_date'] = end_date
                        year_data[fiscal_year]['Fiscal Year'] = fiscal_year
    
    # 결과 데이터프레임 형태로 변환
    for year, metrics in year_data.items():
        if len(metrics) > 2:  # 최소한 회사명과 회계연도 외에 데이터가 있는 경우만 포함
            results.append(metrics)
    
    return results

# 모든 M7 기업의 데이터 수집 및 처리
all_financial_data = []

for company_name, cik in M7_CIK.items():
    company_data = get_company_financial_data(company_name, cik)
    
    if company_data:
        company_metrics = extract_financial_metrics(company_name, company_data)
        all_financial_data.extend(company_metrics)
        print(f"{company_name}의 데이터 처리 완료: {len(company_metrics)}개 회계연도 데이터 추출")
    else:
        print(f"{company_name}의 데이터를 가져오지 못했습니다.")

# 데이터프레임 생성 및 CSV 저장
if all_financial_data:
    # 데이터프레임 생성
    df = pd.DataFrame(all_financial_data)
    
    # 열 순서 정리
    columns = ['Company', 'Fiscal Year']
    metric_columns = [v for v in FINANCIAL_METRICS.values()]
    columns.extend(metric_columns)
    
    # 사용 가능한 열만 선택
    available_columns = [col for col in columns if col in df.columns]
    df = df[available_columns]
    
    # 숫자 데이터를 수백만 달러 단위로 변환 (EPS 제외)
    for col in df.columns:
        if col not in ['Company', 'Fiscal Year', 'end_date', 'EPS Basic', 'EPS Diluted']:
            if col in df.columns:
                try:
                    # 숫자만 변환, 비-숫자 값은 그대로 유지
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                    # EPS가 아닌 값들은 백만 단위로 변환
                    if not col.startswith('EPS'):
                        df[col] = df[col] / 1000000
                except Exception as e:
                    print(f"{col} 열 변환 중 오류: {e}")
    
    # 'end_date' 열 제거 (필요한 경우)
    if 'end_date' in df.columns:
        df = df.drop('end_date', axis=1)
    
    # 회사명과 회계연도별로 정렬
    df = df.sort_values(['Company', 'Fiscal Year'])
    
    # CSV 파일로 저장
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # csv_filename = f"M7_financial_data_{timestamp}.csv"
    # df.to_csv(csv_filename, index=False)
    df.to_csv('../store data/M7_financial_data_2020_2024.csv', index=False)
    
    # print(f"\n성공적으로 데이터를 추출하여 {csv_filename} 파일로 저장했습니다.")
    print(f"총 {len(df)} 개의 회계연도 데이터가 추출되었습니다.")
    print("\n데이터 미리보기:")
    print(df.head())
else:
    print("추출된 데이터가 없습니다.")