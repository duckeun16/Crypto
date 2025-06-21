from binance.client import Client as SpotClient
from binance.um_futures import UMFutures  # USDT-Margined Futures
from binance.client import Client as PublicClient
import pandas as pd
import decimal


# Replace with your actual Binance credentials
API_KEY = 'API_KEY'
API_SECRET = 'API_SECRET'
public_client = PublicClient()  # No keys needed for public endpoints

# Initialize Spot and Futures clients
def get_binance_client(market_type):
    if market_type.upper() == 'SPOT':
        # spot_client
        binance_client = SpotClient(api_key=API_KEY, api_secret=API_SECRET)
        return binance_client
    elif market_type.upper() == 'FUT':
        # futures_client
        binance_client = UMFutures(key=API_KEY, secret=API_SECRET)
        return binance_client
    else:
        raise ValueError(f"Unsupported market_type: {market_type}")

def get_bba_price(market_type, symbol):
    if market_type.upper() == 'SPOT':
        book = public_client.get_order_book(symbol=symbol, limit=5)
    elif market_type.upper() == 'FUT':
        book = public_client.futures_order_book(symbol=symbol, limit=5)
    else:
        raise ValueError(f"Unsupported market_type: {market_type}")
    
    best_bid = float(book['bids'][0][0])
    best_ask = float(book['asks'][0][0])
    return best_bid, best_ask

def get_mid_limit_price(market_type, symbol, side):

    def get_decimal_precision(price: float) -> int:
        """Get the number of decimal places for a float."""
        return abs(decimal.Decimal(str(price)).as_tuple().exponent)
    
    def round_price(price: float, reference_price: float) -> float:
        """Round price using reference price's precision."""
        precision = get_decimal_precision(reference_price)
        return round(price, precision)
        
    best_bid, best_ask = get_bba_price(market_type, symbol)
    raw_mid = (best_bid + best_ask) / 2
    price = round_price(raw_mid, best_bid)  # Use bid's precision as proxy

    # Ensure maker side: use best price if mid would cross
    if side.upper() == 'BUY' and price >= best_ask:
        price = round_price(best_bid, best_bid)
    elif side.upper() == 'SELL' and price <= best_bid:
        price = round_price(best_ask, best_ask)

    print(f"Auto-calculated limit price at mid (if-available): {price}")
    return price, best_bid, best_ask

def cancel_existing_orders(symbol, market_type):
    # cancel all open spot or futures positions for a symbol
    binance_client = get_binance_client(market_type)
    # Cancel existing open limit orders
    if market_type.upper() == 'SPOT':
        open_orders = binance_client.get_open_orders(symbol=symbol)
    else:
        open_orders = binance_client.get_orders(symbol='ETHUSDT')
        
    for order in open_orders:
        if order['type'].upper() == 'LIMIT':
            try:
                binance_client.cancel_order(symbol=symbol, orderId=order['orderId'])
                print(f"Cancelled existing {market_type} limit order {order['orderId']} for {symbol}")
            except Exception as e:
                print(f"Failed to cancel {market_type} order {order['orderId']}: {e}")
    return None

def close_all_positions(symbol, market_type='FUT'):
    # close all futures positions for a symbol (Both long AND short under hedge mode)
    binance_client = get_binance_client(market_type)
    positions = binance_client.get_position_risk()

    for pos in positions:
        if pos['symbol'] == symbol:
            position_amt = float(pos['positionAmt'])
            if position_amt == 0:
                continue
    
            # Determine side to close
            side = 'SELL' if position_amt > 0 else 'BUY'
            # Set positionSide accordingly
            position_side = 'LONG' if position_amt > 0 else 'SHORT'
            qty = abs(position_amt)
    
            print(f"Closing {symbol} position side={side}, positionSide={position_side}, qty={qty}")
            try:
                order = binance_client.new_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=qty,
                    positionSide=position_side  # hedge-mode enabled
                )
                print("Close order response:", order)
            except Exception as e:
                print("Failed to close position:", e)




# === SPOT ORDER FUNCTION ===
def submit_spot_order(symbol, side, quantity, order_type='MARKET', price=None, time_in_force='GTC', cancel_existing=True, market_type='SPOT'):
    try:
        binance_client = get_binance_client(market_type)

        if cancel_existing:
            cancel_existing_orders(symbol, market_type)
        
        if order_type.upper() == 'LIMIT':            
            if price is None:
                # use computed mid price or bba if exists
                price, best_bid, best_ask = get_mid_limit_price(market_type, symbol, side)
            else:
                # Fetch book for post-only
                best_bid, best_ask = get_bba_price(market_type, symbol)
            
            # Final crossing check
            if side.upper() == 'BUY' and price >= best_ask:
                print(f"Reject Spot BUY Limit: price {price} crosses best ask {best_ask}")
                return None
            elif side.upper() == 'SELL' and price <= best_bid:
                print(f"Reject Spot SELL Limit: price {price} crosses best bid {best_bid}")
                return None

        order_params = {
            'symbol': symbol,
            'side': side,
            'type': order_type.upper(),
            'quantity': quantity
        }

        if order_type.upper() == 'LIMIT':
            order_params.update({
                'price': str(price),
                'timeInForce': time_in_force
            })

        response = binance_client.create_order(**order_params)
        print("Spot Order Submitted:", response)
        return response

    except Exception as e:
        print("Error submitting spot order:", e)
        return None


# === FUTURES ORDER FUNCTION ===
def submit_fut_order(
    symbol, side, quantity,
    order_type='MARKET',
    price=None,
    time_in_force='GTC',
    cancel_existing=True,
    market_type='FUT',
    position_side=None  # <-- Optional explicit position side (due to hedge-mode enabled)
):
    try:
        binance_client = get_binance_client(market_type)

        if cancel_existing:
            cancel_existing_orders(symbol, market_type)

        if order_type.upper() == 'LIMIT':
            if price is None:
                # use computed mid price or bba if exists
                price, best_bid, best_ask = get_mid_limit_price(market_type, symbol, side)
            else:
                best_bid, best_ask = get_bba_price(market_type, symbol)

            # Check if price crosses book
            if side.upper() == 'BUY' and price >= best_ask:
                print(f"Reject Futures BUY Limit: price {price} crosses best ask {best_ask}")
                return None
            elif side.upper() == 'SELL' and price <= best_bid:
                print(f"Reject Futures SELL Limit: price {price} crosses best bid {best_bid}")
                return None

        # === Determine position side ===
        if position_side is None:
            position_side = 'LONG' if side.upper() == 'BUY' else 'SHORT'

        # === Order parameters ===
        order_params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity,
            'positionSide': position_side.upper()
        }

        if order_type.upper() == 'LIMIT':
            order_params.update({
                'price': str(price),
                'timeInForce': time_in_force
            })

        response = binance_client.new_order(**order_params)
        print("Futures Order Submitted:", response)
        return response

    except Exception as e:
        print("Error submitting futures order:", e)
        return None


### get balances
def get_spot_balances(market_type='SPOT'):
    try:
        binance_client = get_binance_client(market_type)
        account_info = binance_client.get_account()
        balances = account_info['balances']
        non_zero = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]
        return non_zero
    except Exception as e:
        print("Error fetching spot balances:", e)
        return None

def get_futures_balances(market_type='FUT'):
    try:
        binance_client = get_binance_client(market_type)
        balances = binance_client.balance()
        non_zero = [b for b in balances if float(b['balance']) > 0]
        return non_zero
    except Exception as e:
        print("Error fetching futures balances:", e)
        return None

def compute_execution_delta(order_request):
    
    # transcribe execution order request inputs
    symbol = order_request['symbol']
    target_quantity = order_request['target_quantity']
    target_notional = order_request['target_notional']
    market_type = order_request['market_type']
    tol_pct = order_request['tol_pct']

    if (target_quantity is None and target_notional is None) or \
       (target_quantity is not None and target_notional is not None):
        raise ValueError("Provide exactly one of target_quantity or target_notional")

    # Extract the base asset from symbol, e.g. 'ETH' from 'ETHUSDT'
    # Assuming USDT quote for spot market
    base_asset = ''.join([c for c in symbol if not c.isdigit() and not c.isupper() == False])  # crude but effective
    # More robust:
    if symbol.endswith('USDT'):
        base_asset = symbol[:-4]
    else:
        raise ValueError("This function expects USDT quote symbols")

    # Get current balance of base asset
    balances_df = pd.DataFrame(get_spot_balances())
    asset_balance_row = balances_df[balances_df['asset'] == base_asset]
    if asset_balance_row.empty:
        current_balance = 0.0
    else:
        current_balance = float(asset_balance_row['free'].values[0])

    # Convert target notional to token quantity if needed
    if target_quantity is not None:
        target_qty = target_quantity
    else:
        best_bid, best_ask = get_bba_price(market_type, symbol)
        current_price = (best_bid + best_ask) / 2
        target_qty = target_notional / current_price

    # Calculate how much needs to be bought or sold relative to current balance, rounded to 4th decimal place
    diff = round(target_qty - current_balance, 3)

    # Tolerance level: If difference is zero or small %, no order needed
    if abs(diff) < (target_qty * tol_pct):
        return {'side': None, 'quantity': 0, 'price': None}

    side = 'BUY' if diff > 0 else 'SELL'
    quantity = abs(diff)

    # Get price from provided function (use side as BUY or SELL)
    price, best_bid, best_ask = get_mid_limit_price(market_type, symbol=symbol, side=side)

    return {
        'symbol': symbol, 
        'side': side,
        'quantity': quantity,
        'price': price
    }