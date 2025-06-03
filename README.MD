Yahoo Finance Gap Trader
This Python script implements a gap trading system that identifies trading opportunities based on pre-market price gaps using Yahoo Finance data and executes trades via the Tiger Trade API.
Features
Gap Detection: Identifies significant price gaps at market open.
Configurable Parameters: Easily adjust gap percentages, volume, price thresholds, position sizing, stop-loss, and take-profit levels.
Yahoo Finance Integration: Utilizes yfinance for free and reliable market data.
Tiger Trade API Integration: Connects to your Tiger Trade account for order execution and position monitoring.
Continuous Monitoring: Runs indefinitely, scanning for opportunities at a defined interval.
Logging: Detailed logging to a file and console for monitoring activity.
SQLite Database Storage: Persists identified gap opportunities, executed trades, and current positions to a local SQLite database (trading_data.db).
Prerequisites
Before you begin, ensure you have the following:
Python 3.8+ installed.
Tiger Trade Account with API access enabled.
Tiger Trade API Credentials:
TIGER_ID
TIGER_ACCOUNT (Your Tiger Trade account number)
TIGER_RSA_PRIVATE_KEY_PYTHON (Your RSA private key, including -----BEGIN RSA PRIVATE KEY----- and -----END RSA PRIVATE KEY----- lines, with \n for newlines).
Setup and Installation

1. Clone the Repository (or create files manually)
   If you're setting this up from scratch, create the following files in your project directory:
   main.py (content from OK_Tiger_GAP_Oportunities.py)
   requirements.txt
   .env.example
   README.md (this file)
2. Install Dependencies
   It's recommended to use a virtual environment:
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. Configure Environment Variables
   Create a file named .env in the root directory of your project. Do NOT commit this file to GitHub.
   Copy the contents of .env.example into your new .env file and fill in your actual Tiger Trade API credentials and desired trading parameters.
   Example .env file:
   TIGER_ID="YOUR_TIGER_ID"
   TIGER_ACCOUNT="YOUR_TIGER_ACCOUNT_NUMBER"
   TIGER_RSA_PRIVATE_KEY_PYTHON="-----BEGIN RSA PRIVATE KEY-----\nYOUR_ACTUAL_PRIVATE_KEY_HERE_LINE1\nYOUR_ACTUAL_PRIVATE_KEY_HERE_LINE2\n-----END RSA PRIVATE KEY-----"

# Trading Parameters (adjust as needed)

IS_SANDBOX="True"
MIN_GAP_PERCENT="1.5"
MAX_GAP_PERCENT="15.0"
MIN_VOLUME="50000"
MIN_PRICE="5.0"
POSITION_SIZE="5000"
MAX_POSITIONS="10"
STOP_LOSS_PCT="0.05"
TAKE_PROFIT_PCT="0.08"
SCAN_INTERVAL="600"
MARKET_HOURS_ONLY="False"
LOG_LEVEL="INFO"
API_TIMEOUT="30"

Important Note on TIGER_RSA_PRIVATE_KEY_PYTHON:
Your private key must be a single string. If your private key has multiple lines, replace actual newlines with \n characters within the string, as shown in the example above. 4. Run the Bot
Once configured, you can run the bot from your terminal:
python main.py

The bot will start scanning for gap opportunities, executing trades, and logging its activities to the console and a daily log file (e.g., yahoo_gap_trading_YYYYMMDD.log). All relevant trading data will also be saved to trading_data.db.
To stop the bot, press Ctrl+C. 5. Accessing the SQLite Database
You can inspect the trading_data.db file using any SQLite browser or command-line tool.
Using the sqlite3 command-line tool:
sqlite3 trading_data.db

Once inside the SQLite prompt, you can run SQL queries:
List tables: .tables
Schema of a table: .schema gap_opportunities
Select data: SELECT \* FROM executed_trades;
Deployment on Replit
Replit is an online IDE that allows you to deploy Python applications easily.
Create a New Repl: Go to https://replit.com/new and select "Python" as the template.
Upload Files:
Delete the default main.py file that Replit creates.
Upload your main.py (the renamed OK_Tiger_GAP_Oportunities.py), requirements.txt, and README.md files to your Repl.
Configure Environment Variables (Secrets):
In Replit, do NOT create a .env file directly. Instead, use the "Secrets" tab (lock icon) in the left sidebar.
Add each environment variable (e.g., TIGER_ID, TIGER_ACCOUNT, TIGER_RSA_PRIVATE_KEY_PYTHON, MIN_GAP_PERCENT, etc.) as a new secret.
For TIGER_RSA_PRIVATE_KEY_PYTHON, paste your private key exactly as it is, including newlines. Replit's secrets handling will preserve them.
Run the Repl: Click the "Run" button. Replit will automatically install the dependencies from requirements.txt and then execute main.py. The trading_data.db file will be created in your Repl's file system.
Deployment on GitHub
Initialize a Git Repository:
git init

Create a .gitignore file:
Create a file named .gitignore in your project root and add the following to it:
.env
venv/
\*.log
**pycache**/
trading_data.db # Add this line to ignore the database file

This prevents sensitive information and unnecessary files from being committed.
Add and Commit Files:
git add .
git commit -m "Initial commit of Yahoo Finance Gap Trader with SQLite"

Create a GitHub Repository: Go to GitHub, create a new repository, and follow the instructions to link your local repository to it.
Push to GitHub:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main

Now your project is on GitHub, and you can easily deploy it to platforms like Replit, or other cloud services.
Important Considerations
API Keys and Security: Never expose your private keys or sensitive credentials directly in your code or public repositories. Always use environment variables (or Replit Secrets).
Trading Risks: Automated trading involves significant risks. Understand the strategy, test thoroughly, and use at your own discretion.
Rate Limits: Be mindful of API rate limits for both Yahoo Finance (though yfinance is generally robust) and Tiger Trade.
Market Hours: The MARKET_HOURS_ONLY flag helps restrict trading to active market hours, which is generally recommended.
Error Handling: The script includes basic error handling, but robust production systems would require more comprehensive error management and alerting.
Database Persistence on Replit: On Replit, files created during runtime (like trading_data.db) generally persist across runs within the same Repl. However, if you fork the Repl or if Replit's underlying infrastructure changes, the database file might be reset. For long-term, highly reliable data storage, consider external cloud databases.
# openGapTraderTigerTrader
