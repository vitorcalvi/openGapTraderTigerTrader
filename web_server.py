
from flask import Flask, render_template, send_from_directory
import os
import threading
import time
from main import YahooFinanceGapTrader

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/status')
def status():
    return {"status": "Trading bot is running", "message": "Check console for trading activity"}

def run_trading_bot():
    """Run the trading bot in a separate thread"""
    try:
        required_vars = ['TIGER_ID', 'TIGER_ACCOUNT', 'TIGER_RSA_PRIVATE_KEY_PYTHON']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            return
        
        trader = YahooFinanceGapTrader()
        trader.run_continuous_monitoring()
    except Exception as e:
        print(f"❌ Error in trading bot: {e}")

if __name__ == '__main__':
    # Start the trading bot in a separate thread
    bot_thread = threading.Thread(target=run_trading_bot, daemon=True)
    bot_thread.start()
    
    # Give the bot a moment to start
    time.sleep(2)
    
    # Start the Flask web server
    print("🌐 Starting web server on port 5000...")
    print("📊 Access the dashboard at: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
