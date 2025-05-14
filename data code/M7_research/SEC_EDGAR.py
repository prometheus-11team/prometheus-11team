# TODO: SEC API상 실적발표 결측치 대응
import requests
import pandas as pd
import time

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

# 회계 연도 범위
FISCAL_YEARS = list(range(2020, 2026))

# SEC API 기본 URL
BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"

# 요청 헤더
HEADERS = {
    "User-Agent": "Your Name your@email.com",  # 반드시 실제 정보로 교체
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

# 주요 지표 매핑
FINANCIAL_METRICS = {
    "OperatingIncomeLoss": "Operating Income",
    "NetIncomeLoss": "Net Income",
    "EarningsPerShareDiluted": "EPS Diluted",
    "Assets": "Total Assets",
    "StockholdersEquity": "Shareholders Equity"
}

# 제출일 수집 함수
def get_filings(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        recent_data = data['filings']['recent']

        filings = []
        for i in range(len(recent_data['form'])):
            form_type = recent_data['form'][i]
            filing_date = recent_data['filingDate'][i]
            accession_number = recent_data['accessionNumber'][i]
            filing_link = f"https://www.sec.gov/Archives/{accession_number.replace('-', '')}"

            if form_type in ['10-K', '10-Q']:
                filings.append({
                    "form": form_type,
                    "filing_date": filing_date,
                    "link": filing_link
                })
        return filings
    except Exception as e:
        print(f"❌ Filings 데이터 가져오기 실패: {e}")
        return []

# 회사  데이터 가져오기
def get_company_financial_data(company_name, cik):
    url = BASE_URL.format(cik.zfill(10))
    print(f"{company_name}({cik})의 실적 데이터를 가져오는 중...")
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        time.sleep(0.1)
        return data
    except requests.exceptions.RequestException as e:
        print(f"{company_name}의 데이터 가져오기 실패: {e}")
        return None

# 재무 데이터 및 제출일 통합 처리
def extract_financial_metrics(company_name, data, filing_dates):
    if not data:
        return []
    
    results = []
    company_facts = data.get('facts', {})
    us_gaap = company_facts.get('us-gaap', {})
    period_data = {}
    
    for metric_tag, entries in us_gaap.items():
        display_name = FINANCIAL_METRICS.get(metric_tag)
        if not display_name:
            continue
        
        for unit_type, unit_data in entries.get('units', {}).items():
            for entry in unit_data:
                form_type = entry.get('form')
                if form_type not in ['10-K', '10-Q']:
                    continue
                fiscal_year = entry.get('fy')
                if fiscal_year not in FISCAL_YEARS:
                    continue

                value = entry.get('val')
                end_date = entry.get('end')
                fiscal_period = entry.get('fp')

                period_label = f"{fiscal_year}-Annual" if form_type == '10-K' else f"{fiscal_year}-{fiscal_period}"

                if period_label not in period_data:
                    period_data[period_label] = {
                        "Company": company_name,
                        "Fiscal Year": fiscal_year,
                        "Period": fiscal_period if form_type == '10-Q' else 'Annual',
                        "Report Type": form_type,
                        "End Date": end_date
                    }

                if display_name not in period_data[period_label] or end_date > period_data[period_label].get('End Date', ''):
                    period_data[period_label][display_name] = value
                    period_data[period_label]['End Date'] = end_date

    for period_label, metrics in period_data.items():
        if len(metrics) > 5:
            results.append(metrics)
    
    return results

# 정확한 제출일 매핑 함수
from datetime import datetime
def map_filing_date(row, filings):
    target_form = row['Report Type']
    end_date_str = row['End Date']

    try:
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except Exception as e:
        return None  # 날짜 변환 실패 시 None 반환

    # 대상 양식 필터링
    candidates = [f for f in filings if f['form'] == target_form]

    # 제출일과 End Date 간 차이 계산 후 가장 가까운 제출일 찾기
    closest_filing = None
    min_diff = float('inf')

    for f in candidates:
        try:
            filing_date = datetime.strptime(f['filing_date'], "%Y-%m-%d")
            diff = abs((filing_date - end_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_filing = filing_date
        except Exception:
            continue

    #  시차가 너무 차이날 경우 None 처리 : 4개월(120일)
    if closest_filing and min_diff <= 110:
        return closest_filing.strftime("%Y-%m-%d")
    else:
        return None
    
# 전체 실행
all_financial_data = []

for company_name, cik in M7_CIK.items():
    company_data = get_company_financial_data(company_name, cik)
    filing_dates = get_filings(cik)
    
    if company_data:
        company_metrics = extract_financial_metrics(company_name, company_data, filing_dates)
        all_financial_data.extend(company_metrics)
        print(f"✅ {company_name}의 데이터 처리 완료: {len(company_metrics)}개 보고서 추출")
    else:
        print(f"❌ {company_name} 데이터 처리 실패")

# 결과 저장
if all_financial_data:
    df = pd.DataFrame(all_financial_data)

    columns = ['Company', 'Fiscal Year', 'Period', 'Report Type', 'End Date', 'filing Date']
    metric_columns = [v for v in FINANCIAL_METRICS.values()]
    columns.extend(metric_columns)

    available_columns = [col for col in columns if col in df.columns]
    df = df[available_columns]

    for col in df.columns:
        if col not in ['Company', 'Fiscal Year', 'Period', 'Report Type', 'End Date', 'filing Date', 'EPS Diluted']:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
                if not col.startswith('EPS'):
                    df[col] = df[col] / 1_000_000
            except Exception as e:
                print(f"{col} 변환 오류: {e}")

    df['End Date'] = pd.to_datetime(df['End Date']).dt.strftime('%Y-%m-%d')

    df = df.sort_values(['Company', 'Fiscal Year', 'Period']) # 정렬

    # 기업실적 보고서 제출일 통합
    company_filings_dict = {}
    for company_name, cik in M7_CIK.items(): # filings dict 수집
        filings = get_filings(cik.zfill(10))
        company_filings_dict[company_name] = filings
        time.sleep(0.2)
    print("📅 기업별 실적보고서 제출일 수집 완료! 매핑 중...")

    # 기존 df에 실적발표일(기업의 보고서 제출일) 매핑
    df['Release Date'] = df.apply(lambda row: map_filing_date(row, company_filings_dict.get(row['Company'], [])), axis=1)
    # 날짜 포맷 변환 (매핑된 컬럼에 적용)
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.strftime('%Y-%m-%d')


    df.to_csv('../../store data/M7_financial_data_2020_2025.csv', index=False)
    print(f"\n총 {len(df)} 개의 회계연도 데이터가 저장되었습니다.")
    print("\n📊 데이터 미리보기:")
    print(df.head())
else:
    print("📭 추출된 데이터가 없습니다.")