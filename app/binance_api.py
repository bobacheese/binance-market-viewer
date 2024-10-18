import os
from dotenv import load_dotenv
from binance.client import Client
from binance.streams import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
import numpy as np
import pandas as pd
import ta  # Tambahkan import ini
import logging

load_dotenv()

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_SECRET_KEY'))

logger = logging.getLogger(__name__)

def get_market_data():
    try:
        tickers = client.get_ticker()
        return [
            {
                'symbol': ticker['symbol'],
                'price': float(ticker['lastPrice']),
                'change': float(ticker['priceChangePercent'])
            }
            for ticker in tickers if ticker['symbol'].endswith('USDT')
        ][:10]  # Ambil 10 pair USDT teratas
    except BinanceAPIException as e:
        print(f"Error fetching data from Binance: {e}")
        return []

def start_websocket(socketio):
    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    twm.start()

    def handle_socket_message(msg):
        if msg['e'] == '24hrTicker':
            socketio.emit('price_update', {
                'symbol': msg['s'],
                'price': float(msg['c']),
                'change': float(msg['P'])
            })

    symbols = [ticker['symbol'] for ticker in get_market_data()]
    streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
    twm.start_multiplex_socket(callback=handle_socket_message, streams=streams)

    return twm

def get_futures_symbols():
    exchange_info = client.futures_exchange_info()
    return [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['status'] == 'TRADING']

def get_klines(symbol, interval, limit=1000):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]

def calculate_choppiness(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window).sum()
    highest_high = df['High'].rolling(window).max()
    lowest_low = df['Low'].rolling(window).min()
    choppiness = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(window)
    return choppiness

def calculate_supertrend(df, period=7, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=period).average_true_range()
    
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index)
    direction = pd.Series(index=df.index)
    
    for i in range(period, len(df)):
        if df['Close'].iloc[i] > upperband.iloc[i-1]:
            supertrend.iloc[i] = lowerband.iloc[i]
            direction.iloc[i] = 1
        elif df['Close'].iloc[i] < lowerband.iloc[i-1]:
            supertrend.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
            
        if direction.iloc[i] == 1 and lowerband.iloc[i] < supertrend.iloc[i]:
            supertrend.iloc[i] = lowerband.iloc[i]
        if direction.iloc[i] == -1 and upperband.iloc[i] > supertrend.iloc[i]:
            supertrend.iloc[i] = upperband.iloc[i]
        
        if direction.iloc[i] == 1 and df['Close'].iloc[i] < supertrend.iloc[i]:
            direction.iloc[i] = -1
        elif direction.iloc[i] == -1 and df['Close'].iloc[i] > supertrend.iloc[i]:
            direction.iloc[i] = 1
    
    return supertrend, direction

def calculate_coppock_curve(data, roc1=14, roc2=11, wma_period=10):
    roc1 = data.pct_change(roc1)
    roc2 = data.pct_change(roc2)
    roc_sum = roc1 + roc2
    coppock = roc_sum.rolling(wma_period).mean()
    return coppock

def calculate_bop(open_data, high, low, close):
    return (close - open_data) / (high - low)

def calculate_indicators(df):
    if len(df) < 100:
        logger.warning(f"Data kurang cuy: {len(df)} baris doang")
        return None

    try:
        # Ubah nama kolom
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        indicators = {}
        
        # Indikator yang udah ada
        indicators['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
        indicators['Supertrend'], indicators['Supertrend_Direction'] = calculate_supertrend(df)
        indicators['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()
        indicators['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        indicators['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).chaikin_money_flow()
        indicators['MFI'] = ta.volume.MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).money_flow_index()
        indicators['VWAP'] = ta.volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
        indicators['Stoch'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
        indicators['ROC'] = ta.momentum.ROCIndicator(close=df['Close']).roc()
        indicators['PPO'] = ta.momentum.PercentagePriceOscillator(close=df['Close']).ppo()
        indicators['Keltner_Channel_High'] = ta.volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close']).keltner_channel_hband()
        indicators['Keltner_Channel_Low'] = ta.volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close']).keltner_channel_lband()
        indicators['Donchian_Channel_High'] = ta.volatility.DonchianChannel(high=df['High'], low=df['Low'], close=df['Close']).donchian_channel_hband()
        indicators['Donchian_Channel_Low'] = ta.volatility.DonchianChannel(high=df['High'], low=df['Low'], close=df['Close']).donchian_channel_lband()
        
        # Indikator baru yang lebih advanced
        indicators['Choppiness'] = calculate_choppiness(df)
        aroon_indicator = ta.trend.AroonIndicator(high=df['High'], low=df['Low'])
        indicators['Aroon_Up'] = aroon_indicator.aroon_up()
        indicators['Aroon_Down'] = aroon_indicator.aroon_down()
        indicators['PSARx'] = ta.trend.PSARIndicator(high=df['High'], low=df['Low'], close=df['Close']).psar()
        indicators['DPO'] = ta.trend.DPOIndicator(close=df['Close']).dpo()
        indicators['KST'] = ta.trend.KSTIndicator(close=df['Close']).kst()
        indicators['Ichimoku_A'] = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low']).ichimoku_a()
        indicators['Ichimoku_B'] = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low']).ichimoku_b()
        indicators['Coppock'] = calculate_coppock_curve(df['Close'])
        indicators['Force_Index'] = ta.volume.ForceIndexIndicator(close=df['Close'], volume=df['Volume']).force_index()
        indicators['EOM'] = ta.volume.EaseOfMovementIndicator(high=df['High'], low=df['Low'], volume=df['Volume']).ease_of_movement()
        indicators['BOP'] = calculate_bop(df['Open'], df['High'], df['Low'], df['Close'])
        indicators['UO'] = ta.momentum.UltimateOscillator(high=df['High'], low=df['Low'], close=df['Close']).ultimate_oscillator()
        
        # Gabungin semua indikator ke DataFrame
        df = pd.concat([df, pd.DataFrame(indicators)], axis=1)
        
        logger.debug(f"Indikator udah dihitung bro. Bentuknya: {df.shape}")
        logger.debug(f"Kolom-kolomnya: {df.columns}")
        return df.dropna()
    except Exception as e:
        logger.error(f"Waduh error nih pas ngitung indikator: {e}", exc_info=True)
        return None

def calculate_parabolic_sar(df, step=0.02, max_step=0.2):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    sar = pd.Series(index=df.index)
    trend = pd.Series(index=df.index)
    ep = pd.Series(index=df.index)
    af = pd.Series(index=df.index)

    # Inisialisasi
    trend.iloc[0] = 1
    sar.iloc[0] = low.iloc[0]
    ep.iloc[0] = high.iloc[0]
    af.iloc[0] = step

    # Hitung SAR
    for i in range(1, len(df)):
        if trend.iloc[i-1] > 0:
            sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
        else:
            sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
        
        if trend.iloc[i-1] > 0:
            if low.iloc[i] < sar.iloc[i]:
                trend.iloc[i] = -1
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = low.iloc[i]
                af.iloc[i] = step
            else:
                trend.iloc[i] = 1
                if high.iloc[i] > ep.iloc[i-1]:
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + step, max_step)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
        else:
            if high.iloc[i] > sar.iloc[i]:
                trend.iloc[i] = 1
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = high.iloc[i]
                af.iloc[i] = step
            else:
                trend.iloc[i] = -1
                if low.iloc[i] < ep.iloc[i-1]:
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + step, max_step)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
        
        if trend.iloc[i] > 0:
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i])
        else:
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i])
    
    return sar

# Helper functions for new indicators
def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    return k

def calculate_adx(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    up_move = df['high'] - df['high'].shift()
    down_move = df['low'].shift() - df['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx

def calculate_cci_custom(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

def calculate_roc(close, period=12):
    return ((close / close.shift(period)) - 1) * 100

def calculate_mfi(df, period=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    rmf = tp * df['volume']
    
    positive_money_flow = rmf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
    negative_money_flow = rmf.where(tp < tp.shift(1), 0).rolling(window=period).sum()
    
    money_ratio = positive_money_flow / negative_money_flow
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

def calculate_ichimoku(df, tenkan_period=9, kijun_period=26, senkou_period=52):
    tenkan_sen = (df['high'].rolling(window=tenkan_period).max() + df['low'].rolling(window=tenkan_period).min()) / 2
    kijun_sen = (df['high'].rolling(window=kijun_period).max() + df['low'].rolling(window=kijun_period).min()) / 2
    return tenkan_sen, kijun_sen

def calculate_parabolic_sar(df, step=0.02, max_step=0.2):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    sar = pd.Series(index=df.index)
    trend = pd.Series(index=df.index)
    ep = pd.Series(index=df.index)
    af = pd.Series(index=df.index)

    # Inisialisasi
    trend.iloc[0] = 1
    sar.iloc[0] = low.iloc[0]
    ep.iloc[0] = high.iloc[0]
    af.iloc[0] = step

    # Hitung SAR
    for i in range(1, len(df)):
        if trend.iloc[i-1] > 0:
            sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
        else:
            sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
        
        if trend.iloc[i-1] > 0:
            if low.iloc[i] < sar.iloc[i]:
                trend.iloc[i] = -1
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = low.iloc[i]
                af.iloc[i] = step
            else:
                trend.iloc[i] = 1
                if high.iloc[i] > ep.iloc[i-1]:
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + step, max_step)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
        else:
            if high.iloc[i] > sar.iloc[i]:
                trend.iloc[i] = 1
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = high.iloc[i]
                af.iloc[i] = step
            else:
                trend.iloc[i] = -1
                if low.iloc[i] < ep.iloc[i-1]:
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + step, max_step)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
        
        if trend.iloc[i] > 0:
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i])
        else:
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i])
    
    return sar

def calculate_chaikin_oscillator(df, short_period=3, long_period=10):
    adl = ((2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])) * df['volume']
    adl = adl.cumsum()
    
    chaikin = adl.ewm(span=short_period, adjust=False).mean() - adl.ewm(span=long_period, adjust=False).mean()
    return chaikin

def calculate_awesome_oscillator(df):
    midpoint = (df['high'] + df['low']) / 2
    ao = midpoint.rolling(window=5).mean() - midpoint.rolling(window=34).mean()
    return ao

def calculate_williams_r(df, period=14):
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    wr = (highest_high - df['close']) / (highest_high - lowest_low) * -100
    return wr

def calculate_dpo(df, period=20):
    dpo = df['close'] - df['close'].rolling(window=period).mean().shift(period // 2 + 1)
    return dpo

def calculate_keltner_channel(df, period=20, atr_period=10, multiplier=2):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    atr = calculate_atr(df, atr_period)
    
    kc_middle = typical_price.rolling(window=period).mean()
    kc_upper = kc_middle + multiplier * atr
    kc_lower = kc_middle - multiplier * atr
    
    return kc_upper, kc_lower

def calculate_cmf(df, period=20):
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv * df['volume']
    cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return cmf

def calculate_aroon(df, period=25):
    aroon_up = 100 * df['high'].rolling(window=period+1).apply(lambda x: x.argmax()) / period
    aroon_down = 100 * df['low'].rolling(window=period+1).apply(lambda x: x.argmin()) / period
    return aroon_up, aroon_down

def calculate_vortex(df, period=14):
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift()), 
                    abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    
    vm_plus = abs(df['high'] - df['low'].shift())
    vm_minus = abs(df['low'] - df['high'].shift())
    
    vi_plus = vm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum()
    vi_minus = vm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum()
    
    return vi_plus, vi_minus

def calculate_trix(close, period=15):
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = (ema3 - ema3.shift()) / ema3.shift() * 100
    return trix

def analyze_data(df):
    try:
        # Pastikan DataFrame tidak kosong
        if df.empty:
            logger.warning("DataFrame is empty")
            return None, None, None, None, None

        # Ambil data terakhir
        last_data = df.iloc[-1]

        # Hitung sinyal
        up_signals = 0
        down_signals = 0

        # RSI
        if 'momentum_rsi' in last_data:
            if last_data['momentum_rsi'] < 30:
                up_signals += 1
            elif last_data['momentum_rsi'] > 70:
                down_signals += 1

        # MACD
        if 'MACD' in last_data and 'MACD_Signal' in last_data:
            if last_data['MACD'] > last_data['MACD_Signal']:
                up_signals += 1
            else:
                down_signals += 1

        # Bollinger Bands
        if 'Close' in last_data and 'volatility_bbm' in last_data and 'volatility_bbh' in last_data and 'volatility_bbl' in last_data:
            if last_data['Close'] < last_data['volatility_bbl']:
                up_signals += 1
            elif last_data['Close'] > last_data['volatility_bbh']:
                down_signals += 1

        # Stochastic RSI
        if 'momentum_stoch_rsi' in last_data:
            if last_data['momentum_stoch_rsi'] < 20:
                up_signals += 1
            elif last_data['momentum_stoch_rsi'] > 80:
                down_signals += 1

        # Ichimoku Cloud
        if 'Close' in last_data and 'Ichimoku_SpanA' in last_data and 'Ichimoku_SpanB' in last_data:
            if (last_data['Close'] > last_data['Ichimoku_SpanA']) and (last_data['Close'] > last_data['Ichimoku_SpanB']):
                up_signals += 1
            elif (last_data['Close'] < last_data['Ichimoku_SpanA']) and (last_data['Close'] < last_data['Ichimoku_SpanB']):
                down_signals += 1

        # Determine overall signal
        if up_signals > down_signals:
            signal = 'BUY'
        elif down_signals > up_signals:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'

        # Calculate signal strength
        total_signals = up_signals + down_signals
        if total_signals > 0:
            strength = abs(up_signals - down_signals) / total_signals
        else:
            strength = 0

        max_consensus = max(up_signals, down_signals)

        return signal, strength, up_signals, down_signals, max_consensus
    except Exception as e:
        logger.error(f"Error in analyze_data: {e}", exc_info=True)
        return None, None, None, None, None

def get_market_sentiment(df):
    try:
        last_data = df.iloc[-1]
        rsi = last_data['momentum_rsi']
        
        if rsi < 30:
            return "Oversold"
        elif rsi > 70:
            return "Overbought"
        else:
            return "Neutral"
    except Exception as e:
        logger.error(f"Error in get_market_sentiment: {e}", exc_info=True)
        return "Unknown"

def get_support_resistance(df):
    try:
        # Hitung support dan resistance sederhana
        support = df['Low'].tail(20).min()
        resistance = df['High'].tail(20).max()
        return support, resistance
    except Exception as e:
        logger.error(f"Error in get_support_resistance: {e}", exc_info=True)
        return None, None
