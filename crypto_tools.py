import numpy as np
import pandas as pd

import requests
import yfinance as yf
from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import ccxt
from datetime import datetime

def to_epoch(date_str):
    epoch_date = round(datetime.strptime(str(date_str), '%Y-%m-%d %H:%M:%S').timestamp()*1000)
    return epoch_date

def to_date(epoch_date):
    dt_date = datetime.utcfromtimestamp(round(epoch_date / 1000))
    return dt_date

def plot_missing_pct(df, ax=None):
    missing_pct = 100 * df.isnull().sum(axis=0) / df.shape[0]
    missing_pct.sort_values().plot(kind='barh', ax=ax)

    return missing_pct

def get_top_coins(top_n=30):
    parameters = {
             'vs_currency': 'usd',
             'order': 'market_cap_desc',
             'per_page': 100,
             'page': 1,
             'sparkline': False,
             'locale': 'en'
             }
    coin_market_data = cg.get_coins_markets(**parameters)
    coin_mcap = pd.DataFrame(coin_market_data)
    coin_mcap = coin_mcap.drop(['image', 'high_24h', 'low_24h', 'price_change_24h', 'price_change_percentage_24h',
    'market_cap_change_24h','market_cap_change_percentage_24h', 'ath_date', 'ath_change_percentage',
    'atl_change_percentage', 'atl_date', 'roi'],  axis = 1)
    
    meme_coins = ['doge','shib','pepe']
    stable_coins = ['usdt', 'usdc', 'dai', 'usde', 'usds']
    redundant_coins = ['steth', 'wbtc','weeth','wsteth','weth']
    # filter meme coins and stable coins
    coin_mcap = coin_mcap[~coin_mcap['symbol'].isin(meme_coins + stable_coins + redundant_coins)]
    
    # extract tickers for yf data
    top30_tickers = coin_mcap.sort_values(by='market_cap_rank').head(top_n)['symbol'].str.upper().tolist()
    coin_tickers = [coin_ticker + '-USD' for coin_ticker in top30_tickers]
    coin_labels = [coin_ticker + '_USD' for coin_ticker in top30_tickers]
    
    tickers = ['^GSPC','^DJI','^NDX','USDT-USD','USDC-USD','USDT-KRW','USDC-KRW','KRW=X','MSTR','GLD']
    names = ['SP500','DJI','ND100','USDTUSD','USDCUSD','USDTKRW','USDCKRW','USDKRW','MSTR','GLD']
    tickers = tickers + coin_tickers
    names = names + coin_labels
    ticker_dict = dict(zip(tickers,names))
    print(ticker_dict)
    
    prices = yf.download(tickers, interval='1d', period='max')['Adj Close'].rename(columns=ticker_dict)
    return prices

def long_query(query_func, start_date, end_date, exchange, symbol, defaultType='future', freq_hr=8, max_data_len=1500, mute=True):
    # convert to epoch
    start_date = to_epoch(start_date)
    end_date = to_epoch(end_date)
    time_step = f'{freq_hr}h'
    # convert to datetime
    start_date = to_date(start_date)
    end_date = to_date(end_date)
    period_hrs = (end_date - start_date).days * 24
    num_calls = int(np.ceil((period_hrs / freq_hr) / max_data_len))
    if not mute:
        print(f'start {start_date}, end {end_date}')
        print(period_hrs)
        print(num_calls)
    if not mute:
        print("-"*100)
        print(f"function called: {query_func}")
        print(f'selected timestep: {time_step}')
        print("number of data per query:", max_data_len)
        print("number of queries required:", num_calls)
    query_df_list = []

    for call in range(num_calls):
        end_date = start_date + pd.Timedelta(max_data_len * freq_hr, "hr")
        # to epoch
        start_date = to_epoch(start_date)
        query_df = query_func(exchange, symbol, nobs=max_data_len, start=start_date, freq=time_step, defaultType=defaultType)
        query_df_list.append(query_df)
        # to datetime
        start_date = to_date(start_date)
        if not mute:
            print(f"{call+1}---start: {start_date}, end: {end_date}---")
            print(query_df.index.nunique())
        # set new start time
        start_date = end_date
        
    long_query_df = pd.concat(query_df_list, axis=0)
    # deduplicate long query
    if not mute:
        print('deduplicating timestamps:', long_query_df[long_query_df.duplicated()].index.unique())
        print("-"*100)
    long_query_df = long_query_df[~long_query_df.duplicated()]
    
    return long_query_df

def get_prices(exchange, symbol, nobs, start, defaultType, freq='8h'):
    inst = getattr(ccxt, exchange)({
        'enableRateLimit': True,
        'options': {
            'defaultType': f'{defaultType}',
            'adjustForTimeDifference': True
        }
    })
    try:
        df = pd.DataFrame(inst.fetchOHLCV(symbol,
                                          since=start,
                                          timeframe=freq,
                                          limit=nobs))
    except:
        df = pd.DataFrame(inst.fetchOHLCV(symbol,
                                          since=start,
                                          limit=nobs))
        
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = df.set_index('time').sort_index()
    df.index = pd.to_datetime(df.index, unit='ms').round('60min')
    df = df.astype(float)
    
    return df


def get_funding_rates(exchange, symbol, nobs, start, defaultType='future', freq='8h'):
    inst = getattr(ccxt, exchange)({
        'enableRateLimit': True,
        'options': {
            'defaultType': f'{defaultType}',
            'adjustForTimeDifference': True
        }
    })
    df = pd.DataFrame(inst.fetchFundingRateHistory(symbol=symbol, 
                                                   since=start,
                                                   limit=nobs))
    df = df[['symbol','fundingRate','datetime']]
    df.datetime = pd.to_datetime(df.datetime, utc=True).round('60min')
    df = df.set_index('datetime').sort_index()
    
    return df

def get_cd_crypto_index(start_date_str, end_date_str):
    cd_crypto_index_query_string = f'https://production.api.coindesk.com/v2/tb/price/values/CMI?start_date={start_date_str}T00:00&end_date={end_date_str}T23:59&ohlc=true'
    cd_crypto_index_query_results = requests.get(cd_crypto_index_query_string).json()['data']['entries']
    
    data_df = pd.DataFrame(cd_crypto_index_query_results, columns=['timestamp','open','high','low','close'])
    data_df['timestamp'] = data_df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x/1000)).round('60min')
    print(data_df['timestamp'].min(), data_df['timestamp'].max())
    data_df['index_returns'] = data_df['close'].pct_change()
    data_df = data_df.rename(columns={'timestamp' : 'Date', 
                                      'close' : 'index_close'})
    data_df = data_df.set_index('Date')
    data_df = data_df.shift(-1).ffill() # adjust to match yfinance timing
    
    return data_df


def plot_corr_mat(returns, ax=None):
    corr_mat = returns.dropna().corr()
    
    mask = np.zeros_like(corr_mat, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_mat, cmap='coolwarm', mask=mask, annot=False, cbar=False, ax=ax);
    
    return corr_mat
    
def static_reg(returns, y_asset, X_factors):
    returns = returns[[y_asset] + X_factors].dropna()
    X = returns[X_factors]
    y = returns[y_asset]
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    return model

def rolling_reg(returns, y_asset, X_factors, window):
    returns = returns[[y_asset] + X_factors].dropna()
    X = returns[X_factors]
    y = returns[y_asset]
    model = RollingOLS(y, sm.add_constant(X), window=window, min_nobs=window).fit()
    
    return model

def vectorized_beta(returns, market_definition='ETH'):
    market = returns[market_definition]
    assets = returns
    # Calculate betas for all assets
    market_demeaned = market - market.mean()
    assets_demeaned = assets - assets.mean()
    betas = assets_demeaned.mul(market_demeaned, axis=0).sum(axis=0) / np.sum(market_demeaned ** 2)
    betas.name = market.name
    betas.index.name = 'symbol'

    return betas

def vectorized_corr(returns, market_definition='ETH'):
    # Calculate corrs for all assets
    corrs = returns.corr().loc[:, market_definition]
    corrs.index.name = 'symbol'
    
    return corrs

def filter_outliers(ser, lower_percentile=0.01, upper_percentile=0.99):
    # filter outliers
    lower_threshold,upper_threshold = ser.quantile([lower_percentile, upper_percentile])
    filtered_ser = ser[ser.between(lower_threshold, upper_threshold)]
    return filtered_ser

def standardize(ser):
    return (ser - ser.mean()) / ser.std()
    
def vectorized_rolling_calc(returns, market_definition='ETH', window_size=30, beta=True):
    rolling_list = []
    for w in returns.rolling(window=window_size, min_periods=window_size):
        if w.shape[0] < window_size:
            # make rolling period less than minobs nan
            nan_ser = pd.Series(index=w.columns)
            if beta:
                rolling_list.append(nan_ser)
            else:
                rolling_list.append(nan_ser)
        else:
            # calculate rolling betas
            if beta:
                betas_ser = vectorized_beta(w, market_definition=market_definition)
                rolling_list.append(betas_ser)
            else:
                corrs_ser = vectorized_corr(w, market_definition=market_definition)
                rolling_list.append(corrs_ser)
        
    rolling_df = pd.concat(rolling_list, axis=1).set_axis(returns.index, axis=1).T
    
    return rolling_df

# calculation check
def get_beta_trends(returns_df, market, y_returns, window, plot=False, ax=None):
    ols_model = static_reg(returns_df, y_asset=y_returns, X_factors=[market])
    rols_model = rolling_reg(returns_df, y_asset=y_returns, X_factors=[market], window=window)
    static_corr = returns_df[y_returns].corr(returns_df[market])
    rolling_corr = returns_df[y_returns].rolling(window=window).corr(returns_df[market])
    
    results = {
        'ols_model': ols_model,
        'rols_model': rols_model,
        'static_corr': static_corr,
        'rolling_corr': rolling_corr
    }

    if plot:
        # standardize returns
        standardize(returns_df[y_returns]).plot(ax=ax[0], label=y_returns, c='tab:blue')
        standardize(returns_df[market]).plot(ax=ax[0], label='market', c='tab:orange')
        # Draw horizontal line at y=0
        ax[0].axhline(0, color='black', linestyle='--', linewidth=1)
        
        rols_model.params[market].plot(label=f'{market} {window}d rols_model beta: {rols_model.params[market][-1]:.2f}', ax=ax[1], c='tab:orange')
        rolling_corr.plot(label=f'{market} {window}d rolling_corr: {rolling_corr[-1]:.2f}', ax=ax[1], c='tab:blue')
        ax[1].axhline(ols_model.params[market], label=f'{market} ols_model beta: {ols_model.params[market]:.2f}', ls='--', c='tab:orange')
        ax[1].axhline(static_corr, label=f'{market} static_corr: {static_corr:.2f}', ls='--', c='tab:blue')
        ax[1].axhline(0, c='black')

        for i in range(len(ax)):
            ax[i].axhline(0, color='black')
            ax[i].grid()
            ax[i].legend()
        plt.tight_layout();
    
    return results