import requests
import pandas as pd
import time
# from datetime import datetime

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

# 필터링할 회계 연도 범위 
FISCAL_YEARS = list(range(2020, 2026)) # 2020년부터 2025.03.(1Q)까지

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
    # "Revenues": "Total Revenue", # 결측치 존재하는 칼럼 제거
    # "Revenue": "Total Revenue",  
    # "GrossProfit": "Gross Profit",
    # "EarningsPerShareBasic": "EPS Basic",
    "EarningsPerShareDiluted": "EPS Diluted",
    "Assets": "Total Assets",
    # "Liabilities": "Total Liabilities",
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

# 회사 재무 데이터에서 필요한 지표 추출 (연간 및 분기별)
def extract_financial_metrics(company_name, data):
    if not data:
        return []
    
    results = []
    company_facts = data.get('facts', {})
    us_gaap = company_facts.get('us-gaap', {})
    
    # 연간 및 분기별 데이터를 저장할 딕셔너리
    period_data = {}
    
    # US-GAAP 태그에서 필요한 지표 추출
    for metric_tag, entries in us_gaap.items():
        # 현재 지표가 관심 있는 지표인지 확인
        display_name = FINANCIAL_METRICS.get(metric_tag)
        if not display_name:
            continue
        
        # 단위 데이터 처리 (주로 USD 또는 USD/share)
        for unit_type, unit_data in entries.get('units', {}).items():
            for entry in unit_data:
                # 10-K(연간 보고서) 또는 10-Q(분기 보고서) 데이터만 추출
                if entry.get('form') in ['10-K', '10-Q'] and entry.get('fy') in FISCAL_YEARS:
                    fiscal_year = entry.get('fy')
                    value = entry.get('val')
                    end_date = entry.get('end')
                    form_type = entry.get('form')
                    fiscal_period = entry.get('fp')  # 회계 기간 (예: Q1, Q2, Q3, FY)
                    
                    # 회계 기간에 따른 기간 라벨 생성
                    if form_type == '10-K':
                        period_label = f"{fiscal_year}-Annual"
                    else:
                        # 분기 데이터인 경우 (예: 2020-Q1, 2020-Q2 등)
                        period_label = f"{fiscal_year}-{fiscal_period}"
                    
                    # 해당 기간의 데이터가 없으면 초기화
                    if period_label not in period_data:
                        period_data[period_label] = {
                            "Company": company_name,
                            "Fiscal Year": fiscal_year,
                            "Period": fiscal_period if form_type == '10-Q' else 'Annual',
                            "Report Type": form_type,
                            "End Date": end_date
                        }
                    
                    # 중복 처리: 이미 값이 있으면 가장 최근 filing 데이터 사용
                    current_end_date = period_data[period_label].get('End Date', '')
                    if display_name not in period_data[period_label] or end_date > current_end_date:
                        period_data[period_label][display_name] = value
                        period_data[period_label]['End Date'] = end_date
    
    # 결과 리스트로 변환
    for period_label, metrics in period_data.items():
        if len(metrics) > 5:  # 기본 필드(회사, 회계연도, 기간, 보고서유형, 날짜) 외에 최소 하나 이상의 지표가 있는 경우
            results.append(metrics)
    
    return results

# 모든 M7 기업의 데이터 수집 및 처리
all_financial_data = []

for company_name, cik in M7_CIK.items():
    company_data = get_company_financial_data(company_name, cik)
    
    if company_data:
        company_metrics = extract_financial_metrics(company_name, company_data)
        all_financial_data.extend(company_metrics)
        print(f"✅ {company_name}의 데이터 처리 완료: {len(company_metrics)}개 보고서 데이터 추출")
    else:
        print(f"❌ {company_name}의 데이터를 가져오지 못했습니다.")

# 데이터프레임 생성 및 CSV 저장
if all_financial_data:
    # 데이터프레임 생성
    df = pd.DataFrame(all_financial_data)
    
    # 열 순서 정리
    columns = ['Company', 'Fiscal Year', 'Period', 'Report Type', 'End Date']
    metric_columns = [v for v in FINANCIAL_METRICS.values()]
    columns.extend(metric_columns)
    
    # 사용 가능한 열만 선택
    available_columns = [col for col in columns if col in df.columns]
    df = df[available_columns]
    
    # 숫자 데이터를 수백만 달러 단위로 변환 (EPS 제외)
    for col in df.columns:
        if col not in ['Company', 'Fiscal Year', 'Period', 'Report Type', 'End Date', 'EPS Basic', 'EPS Diluted']:
            if col in df.columns:
                try:
                    # 숫자만 변환, 비-숫자 값은 그대로 유지
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                    # EPS가 아닌 값들은 백만 단위로 변환
                    if not col.startswith('EPS'):
                        df[col] = df[col] / 1000000
                except Exception as e:
                    print(f"{col} 열 변환 중 오류: {e}")
    
    # 날짜 형식 정리
    df['End Date'] = pd.to_datetime(df['End Date']).dt.strftime('%Y-%m-%d')
    
    # 회사명, 회계연도, 기간별로 정렬
    df = df.sort_values(['Company', 'Fiscal Year', 'Period'])
    
    # CSV 파일로 저장
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # csv_filename = f"M7_financial_data_{timestamp}.csv"
    # df.to_csv(csv_filename, index=False)
    df.to_csv('../store data/M7_financial_data_2020_2025.csv', index=False)
    
    # print(f"\n성공적으로 데이터를 추출하여 {csv_filename} 파일로 저장했습니다.")
    print(f"총 {len(df)} 개의 회계연도 데이터가 추출되었습니다.")
    print("\n데이터 미리보기:")
    print(df.head())
else:
    print("추출된 데이터가 없습니다.")
