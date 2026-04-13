import numpy as np

# 전처리 함수 정의
def make_ff_factors(factors, annual_rf=True):
    """
    factors: DataFrame with columns ['KOSPI','SMB','HML','MOM','RF']
    """
    
    df = factors.copy()

    # 0. resampling
    df = df.resample('ME').last()
    
    # 1. 지수 → 수익률 변환
    ret_cols = ['KOSPI','SMB','HML','MOM']
    df[ret_cols] = df[ret_cols].pct_change()
    
    # 2. 무위험금리 변환 (연율 → 일/월 수익률)
    df['RF'] = df['RF'] / 100  # % → 소수화 (예: 3.5% → 0.035)
    df['RF'] = (1 + df['RF']) ** (1/12) - 1
    
    # 4. 컬럼 정리
    df = df[['KOSPI','SMB','HML','MOM','RF']].dropna()
    
    return df

def performance_metrics(portfolio):

    portfolio_NAV    = portfolio['NAV'][1:]
    portfolio_return = portfolio['Return'][1:]
    total_trade      = portfolio['Trade'][1:]
    initial_NAV      = portfolio['NAV'].iloc[0]

    # 1. 연평균 수익률 (CAGR)
    total_months = len(portfolio_return)
    years = total_months / 12
    final_value = portfolio_NAV.iloc[-1]
    CAGR = (final_value / initial_NAV) ** (1 / years) - 1

    # 2. 월간 변동성 (Annualized Volatility)
    vol_monthly = portfolio_return.std()
    vol_annual = vol_monthly * np.sqrt(12)

    # 3. Sharpe Ratio (Rf=0 가정)
    mean_return_monthly = portfolio_return.mean()
    sharpe_ratio = (mean_return_monthly * 12) / vol_annual

    # 4. 최대 낙폭 (MDD)
    cummax_NAV = portfolio_NAV.cummax()
    drawdown = portfolio_NAV / cummax_NAV - 1
    MDD = drawdown.min()

    # 5. 월간 Turnover & 평균 Turnover
    monthly_turnover = total_trade / portfolio_NAV  # NAV 대비 총 거래액
    avg_turnover = monthly_turnover.mean()

    return {
        "CAGR": CAGR,
        "Volatility (ann.)": vol_annual,
        "Sharpe Ratio": sharpe_ratio,
        "MDD": MDD,
        "Average Turnover (monthly)": avg_turnover
    }