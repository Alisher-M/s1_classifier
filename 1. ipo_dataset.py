import os
import re
import html
import time
import pandas as pd
import yfinance as yf
from sec_edgar_downloader import Downloader


# paths
BASE_DIR = r"C:\Users\alish\Desktop\outpeer\4. IPO"

TICKERS_FILE = os.path.join(BASE_DIR, "tickers2425.csv")
SAVE_DIR = os.path.join(BASE_DIR, "s1_filings")
OUTPUT_FILE = os.path.join(BASE_DIR, "ipo_dataset.csv")

# SEC information
COMPANY_NAME = "Alisher Mukhametzhanov"
EMAIL = "alisher.m95@pm.me"


# load tickers
try:
    tickers = pd.read_csv(TICKERS_FILE, encoding="cp1252")
except UnicodeDecodeError:
    tickers = pd.read_csv(TICKERS_FILE, encoding="latin1")

tickers["IPO Date"] = pd.to_datetime(tickers["IPO Date"])
tickers["Symbol"] = tickers["Symbol"].str.strip().str.upper()


# download S-1 filings
dl = Downloader(
    COMPANY_NAME,
    EMAIL,
    download_folder=SAVE_DIR
)

for ticker in tickers["Symbol"]:

    print(f"Downloading S-1 for {ticker}...")

    try:
        dl.get("S-1", ticker)
    except Exception as e:
        print(f"Failed for {ticker}: {e}")

    time.sleep(0.2)


# clean SEC text
def clean_text(text):

    text = re.sub(
        r"<script.*?</script>",
        " ",
        text,
        flags=re.I | re.S
    )

    text = re.sub(
        r"<style.*?</style>",
        " ",
        text,
        flags=re.I | re.S
    )

    text = re.sub(
        r"<!--.*?-->",
        " ",
        text,
        flags=re.S
    )

    text = re.sub(r"<[^>]+>", " ", text)

    text = html.unescape(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# get risk factors
def get_risk_factors(ticker):

    folder = os.path.join(
        SAVE_DIR,
        "sec-edgar-filings",
        ticker,
        "S-1"
    )

    if not os.path.exists(folder):
        return None

    files = []

    for root, dirs, filenames in os.walk(folder):
        for file in filenames:

            if file == "full-submission.txt":
                files.append(
                    os.path.join(root, file)
                )

    if not files:
        return None

    with open(
        files[0],
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        text = f.read()

    text = clean_text(text)

    # find Risk Factors
    match = re.search(
        r"RISK\s+FACTORS(.*?)(USE\s+OF\s+PROCEEDS|DILUTION|CAPITALIZATION|DESCRIPTION\s+OF)",
        text,
        flags=re.I | re.S
    )

    if not match:
        return None

    return match.group(1).strip()


# calculate returns
def get_returns(ticker, ipo_date):

    start = ipo_date - pd.Timedelta(days=3)
    end = ipo_date + pd.Timedelta(days=60)

    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            return None

        close = data["Close"]

        # handle newer yfinance MultiIndex
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna()

        dates = pd.to_datetime(close.index)

        if dates.tz is not None:
            dates = dates.tz_localize(None)

        # first trading day on/after IPO
        positions = dates >= ipo_date

        if not positions.any():
            return None

        first = positions.argmax()

        prices = close.iloc[first:]

        if len(prices) <= 7:
            return None

        ipo_price = float(prices.iloc[0])

        r7 = None
        r14 = None
        r30 = None

        if len(prices) > 7:
            r7 = (
                prices.iloc[7] - ipo_price
            ) / ipo_price * 100

        if len(prices) > 14:
            r14 = (
                prices.iloc[14] - ipo_price
            ) / ipo_price * 100

        if len(prices) > 30:
            r30 = (
                prices.iloc[30] - ipo_price
            ) / ipo_price * 100

        returns = [
            x for x in [r7, r14, r30]
            if x is not None
        ]

        label = "HIT" if any(
            x > 5 for x in returns
        ) else "MISS"

        return r7, r14, r30, label

    except Exception as e:

        print(
            f"Could not get returns for {ticker}: {e}"
        )

        return None


# process everything
results = []

for _, row in tickers.iterrows():

    ticker = row["Symbol"]
    ipo_date = row["IPO Date"]

    print(f"\nProcessing {ticker}...")

    risk_factors = get_risk_factors(ticker)

    if not risk_factors:
        print("No Risk Factors found, skipping.")
        continue

    returns = get_returns(
        ticker,
        ipo_date
    )

    if not returns:
        print("Could not calculate returns, skipping.")
        continue

    r7, r14, r30, label = returns

    results.append({
        "ticker": ticker,
        "ipo_date": ipo_date.strftime("%Y-%m-%d"),
        "return_7d": r7,
        "return_14d": r14,
        "return_30d": r30,
        "label": label,
        "risk_factors": risk_factors
    })

    print(
        f"{ticker}: {label} "
        f"(7d={r7}, 14d={r14}, 30d={r30})"
    )


# save
df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nSaved {len(df)} companies to {OUTPUT_FILE}"
)