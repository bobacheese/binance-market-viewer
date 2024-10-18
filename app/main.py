import os
import logging
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from .binance_api import get_futures_symbols, get_klines, calculate_indicators
from .analysis import analyze_data, get_market_sentiment, get_support_resistance
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_analysis')
def handle_analysis():
    symbols = get_futures_symbols()[:10]  # Batasi ke 10 simbol dulu untuk testing
    total_symbols = len(symbols)
    all_signals = []
    
    for i, symbol in enumerate(symbols):
        progress = int((i + 1) / total_symbols * 100)
        
        try:
            df = get_klines(symbol, '1h', limit=1000)
            logger.debug(f"Got klines for {symbol}. Shape: {df.shape}")
            logger.debug(f"Columns: {df.columns}")
            
            df_with_indicators = calculate_indicators(df)
            
            if df_with_indicators is not None and len(df_with_indicators) > 0:
                logger.debug(f"Calculated indicators for {symbol}. Shape: {df_with_indicators.shape}")
                logger.debug(f"Columns after calculating indicators: {df_with_indicators.columns}")
                
                # Tambahkan logging untuk melihat beberapa baris pertama dari DataFrame
                logger.debug(f"First few rows of DataFrame:\n{df_with_indicators.head()}")
                
                try:
                    signal, strength, up_signals, down_signals, max_consensus = analyze_data(df_with_indicators)
                    sentiment = get_market_sentiment(df_with_indicators)
                    support, resistance = get_support_resistance(df_with_indicators)
                    
                    if signal is not None:
                        all_signals.append({
                            'symbol': symbol,
                            'signal': signal,
                            'strength': strength,
                            'up_signals': up_signals,
                            'down_signals': down_signals,
                            'max_consensus': max_consensus,
                            'sentiment': sentiment,
                            'support': round(float(support), 2) if support is not None else None,
                            'resistance': round(float(resistance), 2) if resistance is not None else None
                        })
                        logger.info(f"Successfully analyzed {symbol}")
                    else:
                        logger.warning(f"Skipping {symbol} due to invalid analysis result")
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            else:
                logger.warning(f"Skipping {symbol} due to insufficient data or calculation error")
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
        
        socketio.emit('analysis_progress', {'progress': progress})
        time.sleep(0.5)  # Tambah delay
    
    logger.info(f"Total signals analyzed: {len(all_signals)}")
    socketio.emit('analysis_complete', {'results': all_signals})

if __name__ == '__main__':
    socketio.run(app, debug=True)
