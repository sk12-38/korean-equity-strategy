import numpy as np

# 전처리 함수 정의
def make_ff_factors(factors, annual_rf=True):
    """
    factors: DataFrame with columns ['KOSPI','SMB','HML','MOM','RF']
    annual_rf=True 이면 RF를 연율(%)로 보고 분기 수익률로 변환
    """

    df = factors.copy()

    # 0. 분기 리샘플링
    df = df.resample('QE').last()

    # 1. 지수 -> 수익률 변환 (분기)
    ret_cols = ['KOSPI', 'SMB', 'HML', 'MOM']
    df[ret_cols] = df[ret_cols].pct_change()

    # 2. 무위험금리 변환
    if annual_rf:
        # 연율 % -> 소수 -> 분기 수익률
        df['RF'] = df['RF'] / 100
        df['RF'] = (1 + df['RF']) ** (1 / 4) - 1
    else:
        # 이미 분기 수익률(%)라고 가정
        df['RF'] = df['RF'] / 100

    # 3. 컬럼 정리
    df = df[['KOSPI', 'SMB', 'HML', 'MOM', 'RF']].dropna()

    return df


def performance_metrics(portfolio):
    portfolio_NAV = portfolio['NAV'][1:]
    portfolio_return = portfolio['Return'][1:]
    total_trade = portfolio['Trade'][1:]
    initial_NAV = portfolio['NAV'].iloc[0]

    periods_per_year = 4  # 분기 데이터

    # 1. 연평균 수익률 (CAGR)
    total_quarters = len(portfolio_return)
    years = total_quarters / periods_per_year
    final_value = portfolio_NAV.iloc[-1]
    CAGR = (final_value / initial_NAV) ** (1 / years) - 1 if years > 0 else np.nan

    # 2. 분기 변동성 (연율화)
    vol_quarterly = portfolio_return.std()
    vol_annual = vol_quarterly * np.sqrt(periods_per_year)

    # 3. Sharpe Ratio (Rf=0 가정)
    mean_return_quarterly = portfolio_return.mean()
    sharpe_ratio = (
        (mean_return_quarterly * periods_per_year) / vol_annual
        if vol_annual != 0
        else np.nan
    )

    # 4. 최대 낙폭 (MDD)
    cummax_NAV = portfolio_NAV.cummax()
    drawdown = portfolio_NAV / cummax_NAV - 1
    MDD = drawdown.min()

    # 5. 분기 Turnover & 평균 Turnover
    quarterly_turnover = total_trade / portfolio_NAV  # NAV 대비 총 거래액
    avg_turnover = quarterly_turnover.mean()

    return {
        "CAGR": CAGR,
        "Volatility (ann.)": vol_annual,
        "Sharpe Ratio": sharpe_ratio,
        "MDD": MDD,
        "Average Turnover (quarterly)": avg_turnover,
    }