import time
import pyupbit as pub
from datetime import datetime, timedelta
import pytz
import requests


def log_in(access_token, secret_token):
    upbit = None
    try:
        upbit = pub.Upbit(access_token, secret_token)
        send_message("Login Success")
    except:
        send_message("Login Fail")
    return upbit


def get_start_time(ticker):
    start_time = None
    try:
        df = pub.get_ohlcv(ticker, interval="day", count=1)
        start_time = df.index[0]
    except:
        send_message('Error occured while retrieving start time')
    return start_time

def get_target_price(ticker, k):
    df = pub.get_ohlcv(ticker, interval="day", count=2)
    target_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return target_price


def check_tp_above_ma(ticker, target_price, ma_period):
    above_ma = False
    try:
        df = pub.get_ohlcv(ticker, interval="day", count=ma_period)
        df['average_price'] = df['close'].shift(1).rolling(ma_period - 1).mean()
        average_price = df.iloc[-1]['average_price']
        above_ma = target_price > average_price
    except:
        send_message('Error occured while checking moving average condition')
    return above_ma



def get_current_price(ticker):
    current_price = None
    try:
        current_price = pub.get_current_price(ticker)
    except:
        send_message('Error occured while retrieving current price')
    return current_price


def send_message(message):
    # initialize variables for telegram bot
    # replace the below BOT_TOKEN with your own BOT_TOKEN (retrieved after creating your own Telegram Bot)
    BOT_TOKEN = "your_bot_token"
    METHOD = 'sendMessage'
    # replace the below CHAT_ID with your own chat_id (retrieved using Telgram Bot Raw)
    CHAT_ID = 'your_chat_id'
    response = requests.get(
        url='https://api.telegram.org/bot{}/{}'.format(BOT_TOKEN, METHOD),
        data={'chat_id': CHAT_ID, 'text': message}
    ).json()
    if not response.get('ok'):
        print('Failed to send the following message at {}:\n\t{}\n'.format(
            datetime.fromtimestamp(response.get('result').get('date')), response.get('result').get('text')))
    else:
        print('Successfully sent the following message at {}:\n\t{}\n'.format(
            datetime.fromtimestamp(response.get('result').get('date')), response.get('result').get('text')))
    return


def main():
    # initialize variables for upbit API
    ACCESS_TOKEN = "your_access_key"
    SECRET_TOKEN = "your_secret_key"
    target_price_updated = False
    tp_above_ma = False
    buy_order_placed = False
    sell_order_placed = False
    buy_price = 0
    # login to upbit
    upbit = log_in(ACCESS_TOKEN, SECRET_TOKEN)
    # initialize variables for volatily breakout strategy
    ticker = 'KRW-BTC'
    k = 0.5
    moving_average_period = 15
    # main algorithm
    while True and upbit:
        try:
            # retrieve current time in KST and drop the timezone
            now = (datetime.now(pytz.timezone('ROK')).replace(tzinfo=None))
            # start at 9:00 KST
            start_time = get_start_time(ticker)
            # end at 9:00 KST (+1)
            end_time = start_time + timedelta(days=1)
            # check for volatility breakout
            if start_time < now < end_time - timedelta(seconds=10):
                # retrieve current price of BTC in KRW
                current_price = get_current_price(ticker)
                if not target_price_updated:
                    # compute target price based on previous daily volatility
                    target_price = get_target_price(ticker, k)
                    # check if target price is above moving average
                    tp_above_ma = check_tp_above_ma(ticker, target_price, moving_average_period)
                    send_message(
                        'Updated target price\n\t\u2192 Ticker : {}\n\t\u2192 Current Price : ₩{:,.0f}\n\t\u2192 Target Price : ₩{:,.0f}\n\t\u2192 Above MA{} : {}'
                        .format(ticker, current_price, target_price, moving_average_period, tp_above_ma))
                    target_price_updated = True
                # buy if the daily volatility breaks out k*100(%)
                if target_price < current_price and tp_above_ma:
                    if not buy_order_placed:
                        # normalize balance by considering 0.05% fee margin
                        normalized_krw_balance = upbit.get_balance('KRW') * 0.9995
                        if normalized_krw_balance > 5000:
                            upbit.buy_market_order(ticker, normalized_krw_balance)
                            send_message(
                                'Placed buy order\n\t\u2192 Ticker : {}\n\t\u2192 Quantity : {:.5f}\n\t\u2192 Price : ₩{:,.0f}'
                                .format(ticker, normalized_krw_balance/current_price, current_price))
                            buy_order_placed = True
                            sell_order_placed = False
                            buy_price = current_price
            # check for remaining balance
            else:
                target_price_updated = False
                tp_above_ma = False
                buy_order_placed = False
                # retrieve current price of BTC in KRW
                current_price = get_current_price(ticker)
                # retrieve current balance of BTC in KRW
                btc_balance = upbit.get_balance('BTC')
                # sell if balance exists
                if not sell_order_placed:
                    if btc_balance * current_price > 5000:
                        # sell all the remaining balance
                        upbit.sell_market_order(ticker, btc_balance)
                        change = (current_price - buy_price)*100.0/buy_price
                        gain_loss = (current_price - buy_price) * btc_balance
                        send_message(
                            'Placed sell order\n\t\u2192 Ticker : {}}\n\t\u2192 Quantity : {:.5f}\n\t\u2192 Price : ₩{:,.0f}\n\t\u2192 Change : {:.2f}%\n\t\u2192 Gain/Loss : {}₩{:,.0f}'
                            .format(ticker, btc_balance, current_price, change, ('+' if gain_loss >= 0 else '-'), abs(gain_loss)))
                        sell_order_placed = True
                        buy_price = 0
            time.sleep(1)
        except Exception as e:
            print(e)
            send_message(
                'Error occurred with the following error message:\n\t\u2192 {}'.format(e))
            time.sleep(1)
    send_message('Terminated crypto bot')
    return


if __name__ == '__main__':
    main()
