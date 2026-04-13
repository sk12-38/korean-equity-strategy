# Korean Equity Strategy Lab

This repository archives Korean equity strategy research notebooks tested during the 2026-1 Quantifi main session.

Raw market and accounting data are not included. Prepare the required local data files before running each notebook.

## Strategies

| Strategy | Folder | Main idea |
|---|---|---|
| ROE | `notebooks/01_ROE` | ROE, PBR, shareholder yield filter |
| CEI | `notebooks/02_CEI` | Composite equity issuance and cost-adjusted backtest |
| Gross Profit | `notebooks/03_GP` | Gross profitability signal |
| Peter Lynch | `notebooks/04_peter_lynch` | EPS/PER and earnings acceleration |
| External Financing | `notebooks/05_external_financing` | Financing activity anomaly |
| EPS + DP | `notebooks/06_eps_dp` | EPS growth, dividend yield, regime analysis |
| Sales / Market Cap | `notebooks/07_sales_per_mkt_cap` | Sales-to-market-cap factor |
| Regime Beta | `notebooks/08_regime_beta` | HMM regime and beta-conditioned portfolio |
| External Debt + Sales | `notebooks/09_external_debt_sales` | Debt financing with sales signal |
| EBITDA / Market Cap | `notebooks/10_EBITDA_to_market_cap` | EBITDA-to-market-cap value/profitability signal |
| Sales - Inventory | `notebooks/11_Sales_minus_inventory` | Sales less inventory factor |
| BAB | `notebooks/13_BAB` | Betting against beta |
| Downside Beta | `notebooks/14_downside_beta` | Downside beta factor |
| JMQ | `notebooks/15_JMQ` | Quality, profitability, growth, and safety factor tests |

## Structure

```text
notebooks/   Strategy research notebooks
src/         Shared helper modules
data/        Data instructions only
requirements.txt
```

## Setup

```powershell
pip install -r requirements.txt
```

## Notes

- CSV, Excel, parquet, pickle, PDF, HTML, zip, and generated output files are excluded from version control.
- Some notebooks expect local Korean equity input files such as adjusted prices, market capitalization, accounting variables, benchmark returns, factor returns, and trading-value data.
- Update file paths inside notebooks if your local data directory differs from the original research workspace.
