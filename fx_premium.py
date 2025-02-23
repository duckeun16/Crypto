import numpy as np
import pandas as pd
import yfinance as yf
import ccxt
import matplotlib.pyplot as plt
from datetime import datetime


def get_prices(exchange, symbol, nobs, start, defaultType, freq='8h'):
    inst = getattr(ccxt, exchange)(
            {
            'enableRateLimit': True,
            'options': {
                'defaultType': f'{defaultType}',
                'adjustForTimeDifference': True
            }
        }
    )
    try:
        df = pd.DataFrame(inst.fetch_ohlcv(symbol,
                                          since=start,
                                          timeframe=freq,
                                          limit=nobs))
    except:
        df = pd.DataFrame(inst.fetch_ohlcv(symbol,
                                          since=start,
                                          limit=nobs))

    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = df.set_index('time').sort_index()
    df.index = pd.to_datetime(df.index, unit='ms').round('60min')
    df = df.astype(float)
    
    return df


def get_yf_prices():    
    tickers = ['USDT-USD','USDC-USD','USDT-KRW','USDC-KRW','KRW=X','SGD=X']
    names = ['USDTUSD','USDCUSD','USDTKRW','USDCKRW','USDKRW','USDSGD']
    tickers = tickers
    names = names
    ticker_dict = dict(zip(tickers,names))    
    prices = yf.download(tickers, interval='1d', period='max', progress=False)['Close'].rename(columns=ticker_dict)
    
    return prices


def rank_days(ser, window=252):
    if window == None:
        return ser.rank(pct=True) * 100
    else:
        return ser.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1]) * 100


def percentile_days(ser, window=252):
    if window == None:
        return ser.apply(lambda x: (ser < x).sum()) / len(ser) * 100
    else:
        return ser.rolling(window).apply(lambda x: sum(x < x.iloc[-1])) / window * 100


def percentile_minmax(ser, window=252):
    if window == None:
        agg_min = ser.min()
        agg_max = ser.max()
        return 100 * (ser - agg_min) / (agg_max - agg_min)
    else:
        rolling_min = ser.rolling(window).min()
        rolling_max = ser.rolling(window).max()
        return 100 * (ser - rolling_min) / (rolling_max - rolling_min)




################################################################################

# main function
def get_fx_premium(regex='^pct_.*_krw$', window=10):
    binance_usdc_usdt_spot = get_prices(exchange='binance', symbol='USDC/USDT', nobs=None, start=None, defaultType='spot', freq='1d')
    upbit_usdt_krw_spot = get_prices(exchange='upbit', symbol='USDT/KRW', nobs=None, start=None, defaultType='spot', freq='1d')
    upbit_usdc_krw_spot = get_prices(exchange='upbit', symbol='USDC/KRW', nobs=None, start=None, defaultType='spot', freq='1d')
    yf_prices = get_yf_prices()
    
    stablecoin_fx_pairs = pd.concat([
        binance_usdc_usdt_spot['close'].rename('binance_usdc_usdt'),
        (1 / binance_usdc_usdt_spot['close']).rename('binance_usdt_usdc'),
        upbit_usdt_krw_spot['close'].rename('upbit_usdt_krw'),
        upbit_usdc_krw_spot['close'].rename('upbit_usdc_krw'),
        yf_prices[['USDTUSD','USDCUSD','USDTKRW','USDCKRW','USDKRW','USDSGD']]
    ], axis=1).ffill()
    
    # value at FX USD notional
    stablecoin_fx_pairs['binance_usdt_usd'] = stablecoin_fx_pairs.binance_usdt_usdc * stablecoin_fx_pairs.USDCUSD
    stablecoin_fx_pairs['binance_usdc_usd'] = stablecoin_fx_pairs.binance_usdc_usdt * stablecoin_fx_pairs.USDTUSD
    stablecoin_fx_pairs['upbit_usdt_usd'] = stablecoin_fx_pairs['upbit_usdt_krw'] / stablecoin_fx_pairs.USDKRW
    stablecoin_fx_pairs['upbit_usdc_usd'] = stablecoin_fx_pairs['upbit_usdc_krw'] / stablecoin_fx_pairs.USDKRW
    stablecoin_fx_pairs['binance_usdt_krw'] = stablecoin_fx_pairs.binance_usdt_usdc * stablecoin_fx_pairs.USDKRW
    stablecoin_fx_pairs['binance_usdc_krw'] = stablecoin_fx_pairs.binance_usdc_usdt * stablecoin_fx_pairs.USDKRW

    # % fx premium in krw notional and usd notional for usdt and usdc
    stablecoin_fx_pairs['pct_usdt_fxprem_krw'] = 100 * (stablecoin_fx_pairs.upbit_usdt_krw / stablecoin_fx_pairs.USDKRW - 1)
    stablecoin_fx_pairs['pct_usdc_fxprem_krw'] = 100 * (stablecoin_fx_pairs.upbit_usdc_krw / stablecoin_fx_pairs.USDKRW - 1)
    stablecoin_fx_pairs['pct_usdt_fxprem_usd'] = 100 * (stablecoin_fx_pairs.upbit_usdt_usd / stablecoin_fx_pairs.USDTUSD - 1)
    stablecoin_fx_pairs['pct_usdc_fxprem_usd'] = 100 * (stablecoin_fx_pairs.upbit_usdc_usd / stablecoin_fx_pairs.USDCUSD - 1)

    # amount fx premium in krw notional and usd notional for usdt and usdc
    stablecoin_fx_pairs['amt_usdt_fxprem_krw'] = stablecoin_fx_pairs.upbit_usdt_krw - stablecoin_fx_pairs.USDKRW 
    stablecoin_fx_pairs['amt_usdc_fxprem_krw'] = stablecoin_fx_pairs.upbit_usdc_krw - stablecoin_fx_pairs.USDKRW 
    stablecoin_fx_pairs['amt_usdt_fxprem_usd'] = stablecoin_fx_pairs.upbit_usdt_usd - stablecoin_fx_pairs.USDTUSD
    stablecoin_fx_pairs['amt_usdc_fxprem_usd'] = stablecoin_fx_pairs.upbit_usdc_usd - stablecoin_fx_pairs.USDCUSD

    # % kimchi premium in krw notional and usd notional for usdt and usdc
    stablecoin_fx_pairs['pct_usdt_kimprem_usd'] = 100 * (stablecoin_fx_pairs.upbit_usdt_usd / stablecoin_fx_pairs.binance_usdt_usd - 1)
    stablecoin_fx_pairs['pct_usdc_kimprem_usd'] = 100 * (stablecoin_fx_pairs.upbit_usdc_usd / stablecoin_fx_pairs.binance_usdc_usd - 1)
    stablecoin_fx_pairs['pct_usdt_kimprem_krw'] = 100 * (stablecoin_fx_pairs.upbit_usdt_krw / stablecoin_fx_pairs.binance_usdt_krw - 1)
    stablecoin_fx_pairs['pct_usdc_kimprem_krw'] = 100 * (stablecoin_fx_pairs.upbit_usdc_krw / stablecoin_fx_pairs.binance_usdc_krw - 1)

    # amount kimchi premium in krw notional and usd notional for usdt and usdc
    stablecoin_fx_pairs['amt_usdt_kimprem_usd'] = stablecoin_fx_pairs.upbit_usdt_usd - stablecoin_fx_pairs.binance_usdt_usd
    stablecoin_fx_pairs['amt_usdc_kimprem_usd'] = stablecoin_fx_pairs.upbit_usdc_usd - stablecoin_fx_pairs.binance_usdc_usd
    stablecoin_fx_pairs['amt_usdt_kimprem_krw'] = stablecoin_fx_pairs.upbit_usdt_krw - stablecoin_fx_pairs.binance_usdt_krw
    stablecoin_fx_pairs['amt_usdc_kimprem_krw'] = stablecoin_fx_pairs.upbit_usdc_krw - stablecoin_fx_pairs.binance_usdc_krw

    # % kimchi premium in krw notional and usd notional for usdc and usdt using yf price
    stablecoin_fx_pairs['yf_pct_kimprem_usdc_krw'] = 100 * (stablecoin_fx_pairs.USDCKRW / (stablecoin_fx_pairs.USDCUSD * stablecoin_fx_pairs.USDKRW) - 1)
    stablecoin_fx_pairs['yf_pct_kimprem_usdt_krw'] = 100 * (stablecoin_fx_pairs.USDTKRW / (stablecoin_fx_pairs.USDTUSD * stablecoin_fx_pairs.USDKRW) - 1)
    stablecoin_fx_pairs['yf_pct_kimprem_usdc_usd'] = 100 * ((stablecoin_fx_pairs.USDCKRW / stablecoin_fx_pairs.USDKRW) / stablecoin_fx_pairs.USDCUSD - 1)
    stablecoin_fx_pairs['yf_pct_kimprem_usdt_usd'] = 100 * ((stablecoin_fx_pairs.USDTKRW / stablecoin_fx_pairs.USDKRW) / stablecoin_fx_pairs.USDTUSD - 1)
    
    # amount kimchi premium in krw notional and usd notional for usdc and usdt using yf price
    stablecoin_fx_pairs['yf_amt_kimprem_usdc_krw'] = stablecoin_fx_pairs.USDCKRW - (stablecoin_fx_pairs.USDCUSD * stablecoin_fx_pairs.USDKRW)
    stablecoin_fx_pairs['yf_amt_kimprem_usdt_krw'] = stablecoin_fx_pairs.USDTKRW - (stablecoin_fx_pairs.USDTUSD * stablecoin_fx_pairs.USDKRW)
    stablecoin_fx_pairs['yf_amt_kimprem_usdc_usd'] = (stablecoin_fx_pairs.USDCKRW / stablecoin_fx_pairs.USDKRW) - stablecoin_fx_pairs.USDCUSD
    stablecoin_fx_pairs['yf_amt_kimprem_usdt_usd'] = (stablecoin_fx_pairs.USDTKRW / stablecoin_fx_pairs.USDKRW) - stablecoin_fx_pairs.USDTUSD
    stablecoin_fx_pairs = stablecoin_fx_pairs.dropna()

    summ_ser = stablecoin_fx_pairs.filter(regex=regex).mean(axis=1)
    rank_df = pd.DataFrame({
        # rolling ranks
        'rolling_rank_days' : rank_days(summ_ser, window=window),
        'rolling_percentile_days' : percentile_days(summ_ser, window=window),
        'rolling_percentile_minmax' : percentile_minmax(summ_ser, window=window),
        # global ranks
        'global_rank_days' : rank_days(summ_ser, window=None),
        'global_percentile_days' : percentile_days(summ_ser, window=None),
        'global_percentile_minmax' : percentile_minmax(summ_ser, window=None)
    })

    output_dict = {
        'stablecoin_fx_pairs': stablecoin_fx_pairs,
        'rank': rank_df,
        'summ_ser': summ_ser,
        'summ_rank': rank_df.mean(axis=1)
        
    }

    binance_swap_rate = stablecoin_fx_pairs.binance_usdc_usdt.iloc[-1]
    upbit_swap_rate = stablecoin_fx_pairs.upbit_usdt_krw.iloc[-1]
    final_swap_rate = binance_swap_rate * upbit_swap_rate
    fx_swap_rate = stablecoin_fx_pairs.USDKRW.iloc[-1]
    print(f'binance USDC/USDT: 1 USDC: {binance_swap_rate :0.4f} USDT')
    print(f'upbit USDT/KRW: 1 USDT: {upbit_swap_rate :0.4f} KRW')
    print(f'USDC > binance USDT > upbit KRW swap rate: 1 USDC: {final_swap_rate :0.4f} KRW')
    print(f'fx USD/KRW: 1 USD: {fx_swap_rate :0.4f} KRW')
    print(f'fx premium amt: 1 USDC: {final_swap_rate - fx_swap_rate :0.4f} KRW (final premium: {100 * (final_swap_rate/fx_swap_rate - 1) :0.2f}%)')

    print(f'\n% avg premium: {summ_ser.iloc[-1] :0.2f}%')
    print(f'% avg rank: {rank_df.mean(axis=1).iloc[-1] :0.2f} / 100%')
    
    return output_dict