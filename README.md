# Korean Equity Strategy

## 한국어

이 저장소는 2026년 1학기 Quantifi 메인 세션에서 실험한 한국 주식시장 전략 연구 노트북을 정리한 공간입니다. 원천 시장 데이터와 회계 데이터는 포함하지 않았으므로, 각 노트북을 실행하려면 필요한 로컬 데이터 파일을 준비해야 합니다.

### 전략 목록

| 전략 | 폴더 | 핵심 아이디어 |
| --- | --- | --- |
| ROE | `notebooks/01_ROE` | ROE, PBR, 주주환원 필터 |
| CEI | `notebooks/02_CEI` | Composite equity issuance 및 거래비용 반영 백테스트 |
| Gross Profit | `notebooks/03_GP` | 매출총이익 기반 수익성 신호 |
| Peter Lynch | `notebooks/04_peter_lynch` | EPS/PER 및 이익 증가율 |
| External Financing | `notebooks/05_external_financing` | 외부자금조달 anomaly |
| EPS + DP | `notebooks/06_eps_dp` | EPS 성장률, 배당수익률, 국면 분석 |
| Sales / Market Cap | `notebooks/07_sales_per_mkt_cap` | 매출액 대비 시가총액 factor |
| Regime Beta | `notebooks/08_regime_beta` | HMM 국면 및 beta 조건부 포트폴리오 |
| External Debt + Sales | `notebooks/09_external_debt_sales` | 부채 조달과 매출 신호 결합 |
| EBITDA / Market Cap | `notebooks/10_EBITDA_to_market_cap` | EBITDA 대비 시가총액 가치/수익성 신호 |
| Sales - Inventory | `notebooks/11_Sales_minus_inventory` | 매출액에서 재고를 차감한 factor |
| BAB | `notebooks/13_BAB` | Betting Against Beta |
| Downside Beta | `notebooks/14_downside_beta` | 하방 beta factor |
| JMQ | `notebooks/15_JMQ` | 품질, 수익성, 성장성, 안정성 factor 테스트 |

### 구조

```text
notebooks/   전략 연구 노트북
src/         공통 helper module
data/        데이터 준비 안내
requirements.txt
```

### 실행 환경

```powershell
pip install -r requirements.txt
```

### 참고

- CSV, Excel, parquet, pickle, PDF, HTML, zip, 생성 결과 파일은 버전 관리에서 제외했습니다.
- 일부 노트북은 수정주가, 시가총액, 회계 변수, 벤치마크 수익률, factor 수익률, 거래대금 데이터 등 로컬 한국 주식시장 데이터를 필요로 합니다.
- 로컬 데이터 경로가 다르면 노트북 내부의 파일 경로를 수정해야 합니다.

## English

This repository archives Korean equity strategy research notebooks tested during the 2026-1 Quantifi main session. Raw market and accounting data are not included, so the required local data files must be prepared before running each notebook.

### Strategies

| Strategy | Folder | Main Idea |
| --- | --- | --- |
| ROE | `notebooks/01_ROE` | ROE, PBR, and shareholder yield filter |
| CEI | `notebooks/02_CEI` | Composite equity issuance and cost-adjusted backtest |
| Gross Profit | `notebooks/03_GP` | Gross profitability signal |
| Peter Lynch | `notebooks/04_peter_lynch` | EPS/PER and earnings acceleration |
| External Financing | `notebooks/05_external_financing` | Financing activity anomaly |
| EPS + DP | `notebooks/06_eps_dp` | EPS growth, dividend yield, and regime analysis |
| Sales / Market Cap | `notebooks/07_sales_per_mkt_cap` | Sales-to-market-cap factor |
| Regime Beta | `notebooks/08_regime_beta` | HMM regime and beta-conditioned portfolio |
| External Debt + Sales | `notebooks/09_external_debt_sales` | Debt financing combined with a sales signal |
| EBITDA / Market Cap | `notebooks/10_EBITDA_to_market_cap` | EBITDA-to-market-cap value/profitability signal |
| Sales - Inventory | `notebooks/11_Sales_minus_inventory` | Sales less inventory factor |
| BAB | `notebooks/13_BAB` | Betting Against Beta |
| Downside Beta | `notebooks/14_downside_beta` | Downside beta factor |
| JMQ | `notebooks/15_JMQ` | Quality, profitability, growth, and safety factor tests |

### Structure

```text
notebooks/   Strategy research notebooks
src/         Shared helper modules
data/        Data instructions
requirements.txt
```

### Setup

```powershell
pip install -r requirements.txt
```

### Notes

- CSV, Excel, parquet, pickle, PDF, HTML, zip, and generated output files are excluded from version control.
- Some notebooks expect local Korean equity input files such as adjusted prices, market capitalization, accounting variables, benchmark returns, factor returns, and trading-value data.
- Update file paths inside notebooks if your local data directory differs from the original research workspace.
