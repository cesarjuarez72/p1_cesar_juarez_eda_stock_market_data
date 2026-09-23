# US Equities Historical Exploratory Data Analysis (1970–2018)

An end-to-end data audit, forensic cleaning, and exploratory data analysis (EDA) pipeline evaluating 20,973,889 historical daily trading records across NASDAQ and NYSE. This repository accompanies the formal technical report and Google Colab notebook, establishing a validated data engineering foundation for downstream quantitative finance and machine learning applications.

---

## Project Overview

Machine learning models and quantitative trading algorithms inherit the integrity—or flaws—of their underlying data. This project analyzes a multi-decade equities dataset covering 48.6 years of daily market records. 

Rather than relying on superficial descriptive summaries, the analysis applies an investigative workflow (**Detect $\rightarrow$ Investigate $\rightarrow$ Document $\rightarrow$ Decide**) to trace the origins of structural nulls, corporate split adjustments, and data pipeline corruption artifacts.

### Key Dataset Metrics
* **Total Trading Records:** 20,973,889 rows
* **Securities Tracked:** 6,459 active and historical listings (after cleaning)
* **Temporal Horizon:** January 2, 1970 – August 24, 2018 (17,766 calendar days)
* **Exchanges Covered:** NASDAQ (3,308 securities; 51.22%) and NYSE (3,151 securities; 48.78%)
* **Primary Features:** `ticker`, `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`

---

## Repository Structure

```text
├── data/
│   ├── raw/
│   │   ├── historical_stock_prices.csv   # 20.9M row transactional trade feed
│   │   └── historical_stocks.csv         # Securities metadata catalog
│   └── processed/                        # Cleaned data checkpoints
├── notebooks/
│   └── US_Stock_Market_EDA.ipynb         # Full 8-part executable Colab notebook
├── reports/
│   ├── technical_report.pdf              # Comprehensive technical report
│   └── figures/
│       ├── categorical_distributions.png # Exchange and sector bar charts
│       ├── price_volume_distributions.png# Log-scaled histograms and boxplots
│       └── aapl_historical_trajectory.png# Dual-panel AAPL longitudinal case study
├── src/
│   ├── data_loader.py                    # Schema enforcement and ingestion scripts
│   └── visualizers.py                    # Standardized plotting utilities
├── README.md                             # Repository documentation
└── requirements.txt                      # Python environment dependencies
```

---

## 8-Part Analytical Workflow

The pipeline is implemented across eight structured modules within the Jupyter/Colab notebook:

1. **Environment Setup & Data Ingestion:** Memory-efficient loading of `df_prices` (~21M rows) and `df_stocks` (~6.5K rows) with explicit data type casting.
2. **Structural & Schema Audits:** Identifying null values, duplicate entries, schema alignment issues, and logical pricing corridors (`high >= low`, `open`/`close` within boundaries).
3. **Date Parsing & Temporal Boundaries:** Converting string dates into native `datetime64[ns]` format and mapping historical start/end date constraints.
4. **Categorical Exploration:** Auditing market share distributions between NASDAQ and the NYSE, along with company frequency across 12 GICS economic sectors.
5. **Numerical Distribution & Skewness Analysis:** Evaluating central tendency and dispersion metrics across `open`, `high`, `low`, `close`, and `volume`.
6. **Forensic Outlier Investigations:** Isolating extreme price records ($>1,000,000\text{ USD}$) and adjusted close anomalies to uncover upstream calculation issues.
7. **Distribution Visualizations:** Generating log-transformed histograms and sampled boxplots to expose the underlying log-normal distributions of asset prices and share volumes.
8. **Longitudinal Single-Stock Deep-Dive (Apple Inc. - `AAPL`):** Tracing 37.7 years of trading history (1980–2018), evaluating corporate actions (such as the 2014 7-for-1 split), and formulating predictive research hypotheses.

---

## Key Findings & Data Engineering Decisions

### 1. Injected Header Row Purge
* **Issue:** Row index 74 in `df_stocks` contained the raw header string `['Symbol', 'NYSE', 'NAME', 'SECTOR', 'INDUSTRY']`, caused by a dirty concatenation during upstream file merging.
* **Action:** Purged row 74, preventing string pollution in foreign-key ticker joins and categorical indexes.

### 2. Treatment of Unclassified Sectors
* **Issue:** 1,440 securities (22.29%) lacked sector and industry classifications.
* **Resolution:** Preserved all rows and imputed the category `'Unknown'`. Dropping these tickers would introduce survivorship bias by removing Exchange-Traded Funds (ETFs), index trusts, and SPAC shell entities that trade like equities but do not produce commercial goods.

### 3. Mean vs. Median Divergence (Right-Skewness)
* **Statistical Finding:** The mean closing price of the market sits at **76.11 USD**, while the median closing price is **15.45 USD**. The mean is inflated by $4.93\times$ relative to the median.
* **Analytical Takeaway:** Stock prices follow a strongly right-skewed, log-normal distribution. Over **75% of all historical market sessions closed under 29.72 USD**. Using simple arithmetic averages to characterize typical stock prices yields misleading conclusions; median baselines and logarithmic scaling are required.

| Feature | Mean | Median (50%) | 25th % | 75th % | Max Recorded |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Close** | 76.11 USD | 15.45 USD | 7.50 USD | 29.72 USD | 1,779,750.00 USD |
| **Volume** | 1,227,043 shares | 126,000 shares | 22,100 shares | 607,400 shares | 4,483,504,400 shares |

### 4. Reverse Stock Splits vs. Upstream Floating-Point Overflows
* **Raw Price Artifacts (`SCON`, `CODA`, `TOPS`):** Securities registered raw closing prices between 1.1M USD and 1.78M USD on volumes of fewer than 400 shares. These were traced to micro-cap companies executing repeated reverse stock splits (e.g., 1-for-50) to avoid delisting. Unadjusted historical reconstructions mathematically compounded earlier share prices. These rows reflect legitimate corporate capital mechanics and were retained.
* **Calculation Overflow (`AAN`):** Aaron's Inc. traded normally on February 27, 1987 at **1.30 USD**, but displayed an `adj_close` of **18.9 quintillion USD** ($18,949,618,910,812,438,528.00\text{ USD}$)—a $14.6\text{ trillion}\times$ multiplication error. Caused by an unhandled divide-by-zero or missing split divisor in the vendor pipeline, this entry highlights why raw prices must be inspected before training return-based models.

---

## Getting Started

### Prerequisites
* Python 3.10 or higher
* Recommended: 12GB+ RAM runtime (Google Colab Standard or Pro tier)

### Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/us-stock-market-eda.git](https://github.com/yourusername/us-stock-market-eda.git)
   cd us-stock-market-eda
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter or open the notebook directly in Google Colab:
   ```bash
   jupyter lab notebooks/US_Stock_Market_EDA.ipynb
   ```

---

## Future Research Hypotheses

1. **Volume Surges as Forward Volatility Triggers:** Does an anomalous daily volume surge exceeding $+3.0\sigma$ above its 30-day moving average reliably predict elevated 20-day forward price volatility across large-cap equities?
2. **Defensive Sector Resilience During Systematic Contractions:** Did non-cyclical sectors (Health Care, Consumer Non-Durables) maintain statistically lower maximum drawdowns and superior risk-adjusted Sharpe ratios compared to cyclical sectors during the 2000 Dot-com and 2008 Financial crises?

---

## Author & Academic Context

* **Author:** Cesar Juarez
* **Program:** Data Analytics & Artificial Intelligence
* **Institution:** Willis College, Ottawa, ON
* **Project:** Exploratory Data Analysis & Technical Audit of US Financial Equities
