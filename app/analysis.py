import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def analyze_data(df):
    try:
        signals = []

        # Indikator yang udah ada
        if 'momentum_rsi' in df.columns:
            signals.append(1 if df['momentum_rsi'].iloc[-1] < 30 else -1 if df['momentum_rsi'].iloc[-1] > 70 else 0)
        
        # ... (logika untuk indikator yang udah ada tetap sama)

        # Logika untuk indikator baru
        if 'Choppiness' in df.columns:
            signals.append(0 if df['Choppiness'].iloc[-1] > 61.8 else 1 if df['Choppiness'].iloc[-1] < 38.2 else 0)
        
        if 'Aroon_Up' in df.columns and 'Aroon_Down' in df.columns:
            signals.append(1 if df['Aroon_Up'].iloc[-1] > df['Aroon_Down'].iloc[-1] else -1)
        
        if 'PSARx' in df.columns and 'Close' in df.columns:
            signals.append(1 if df['Close'].iloc[-1] > df['PSARx'].iloc[-1] else -1)
        
        if 'DPO' in df.columns:
            signals.append(1 if df['DPO'].iloc[-1] > 0 else -1)
        
        if 'KST' in df.columns:
            signals.append(1 if df['KST'].iloc[-1] > 0 else -1)
        
        if 'Ichimoku_A' in df.columns and 'Ichimoku_B' in df.columns and 'Close' in df.columns:
            signals.append(1 if df['Close'].iloc[-1] > df['Ichimoku_A'].iloc[-1] and df['Close'].iloc[-1] > df['Ichimoku_B'].iloc[-1] else -1)
        
        if 'Coppock' in df.columns:
            signals.append(1 if df['Coppock'].iloc[-1] > 0 else -1)
        
        if 'Force_Index' in df.columns:
            signals.append(1 if df['Force_Index'].iloc[-1] > 0 else -1)
        
        if 'EOM' in df.columns:
            signals.append(1 if df['EOM'].iloc[-1] > 0 else -1)
        
        if 'BOP' in df.columns:
            signals.append(1 if df['BOP'].iloc[-1] > 0 else -1)
        
        if 'UO' in df.columns:
            signals.append(1 if df['UO'].iloc[-1] > 70 else -1 if df['UO'].iloc[-1] < 30 else 0)

        up_signals = sum(1 for s in signals if s > 0)
        down_signals = sum(1 for s in signals if s < 0)

        if up_signals > down_signals:
            signal = 'BUY'
        elif down_signals > up_signals:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'

        total_signals = len(signals)
        strength = abs(up_signals - down_signals) / total_signals if total_signals > 0 else 0
        max_consensus = max(up_signals, down_signals)

        return signal, strength, up_signals, down_signals, max_consensus
    except Exception as e:
        logger.error(f"Waduh error nih pas analisis data: {e}", exc_info=True)
        return None, None, None, None, None

def get_market_sentiment(df):
    try:
        if 'momentum_rsi' in df.columns:
            rsi = df['momentum_rsi'].iloc[-1]
            
            if rsi < 30:
                return "Oversold"
            elif rsi > 70:
                return "Overbought"
            else:
                return "Neutral"
        else:
            logger.warning("RSI gak ketemu di DataFrame")
            return "Unknown"
    except Exception as e:
        logger.error(f"Error pas ngecek sentiment pasar: {e}", exc_info=True)
        return "Unknown"

def get_support_resistance(df):
    try:
        if 'Low' in df.columns and 'High' in df.columns:
            support = df['Low'].tail(20).min()
            resistance = df['High'].tail(20).max()
            return support, resistance
        else:
            logger.warning("Kolom 'Low' atau 'High' gak ada di DataFrame")
            return None, None
    except Exception as e:
        logger.error(f"Error pas ngitung support resistance: {e}", exc_info=True)
        return None, None

# ... (fungsi-fungsi lainnya tetap sama)
