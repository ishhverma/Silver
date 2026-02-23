#  Silver Volatility Modeling & Options Analytics
### End-to-End Quantitative Research & Trading System
This project develops a **quantitative finance framework** for modeling, forecasting, and trading volatility in silver futures markets. It combines **econometric models, machine learning, deep learning, derivatives pricing, and portfolio risk management** into a unified research pipeline.

#  Project Objectives
The primary goals of this project were:
* Forecast future volatility using statistical and ML techniques
* Price derivatives under multiple volatility assumptions
* Quantify portfolio risk using VaR and Expected Shortfall
* Design and backtest systematic trading strategies
* Integrate macroeconomic factors into predictive models
* Build a scalable quantitative research pipeline

#  Key Skills Demonstrated
✔ Time Series Modeling (GARCH Family, EWMA)
✔ Machine Learning (Random Forest, XGBoost, LightGBM)
✔ Deep Learning (LSTM, GRU)
✔ Options Pricing (Black-Scholes, Monte Carlo, Heston)
✔ Risk Management (VaR, ES, Stress Testing)
✔ Feature Engineering for Financial Data
✔ Quantitative Trading Strategy Development
✔ Portfolio Optimization
✔ Statistical Testing & Model Evaluation
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/50d48453-51e3-4320-b90d-f37f38f9b55e" />

#  Data Overview
**Asset:** Silver Futures (SI=F)
**Period:** 2000-01-01 → 2026-02-22
**Observations:** 6,395 daily records

Macro factors integrated:
* Gold prices
* Dollar index
* Interest rates
* VIX volatility index
* S&P 500
Final modeling dataset:
> **6,394 observations | 38 engineered features**

#  Statistical Characteristics of Returns
| Metric   | Value    |
| -------- | -------- |
| Mean     | 0.000449 |
| Std Dev  | 0.020857 |
| Skewness | -1.547   |
| Kurtosis | 22.43    |
| Min      | -0.376   |
| Max      | 0.131    |

Jarque-Bera Test:
> p-value ≈ 0 → Returns are **non-normal with fat tails**, confirming the need for advanced volatility models.

# Feature Engineering Pipeline
The system generated a rich set of predictive variables:
### Volatility Features
* 7-day, 30-day, 90-day realized volatility
* Volatility-of-volatility
* Jump intensity metrics

### Technical Indicators
* RSI (0.78 → 98.15 range)
* MACD
* Moving averages (20, 50, 200)
* Momentum indicators (5, 20, 60 days)
  
### Lag Structure
Returns lags: 1, 2, 3, 5, 10 days

## Jump Detection
* 325 jumps detected
* 5.08% jump frequency
  <img width="1238" height="528" alt="image" src="https://github.com/user-attachments/assets/bc515e7e-1ff7-4db6-b26c-a4dfdece18af" />

### Macro Relationships
Strongest correlations with silver returns:
* Gold: **0.777**
* Dollar Index: **−0.357**
* VIX: −0.099

#  Volatility Modeling — Econometric Methods
## EWMA
* Lambda: 0.94
* Current volatility: **1.2792**
* RMSE: **0.0334**
## GARCH(1,1)
* 5-day forecast: **1.2406% → 1.2354%**
* RMSE: **0.0376**
## EGARCH / GJR-GARCH
* Asymmetry parameter: −0.0316
* RMSE: 0.0500
Insight:
> EWMA performed best among traditional models for short-horizon forecasting.
<img width="1489" height="590" alt="image" src="https://github.com/user-attachments/assets/5b3e243b-3f26-4b6e-b42a-c19be625f5a5" />

#  Machine Learning Volatility Forecasting
Dataset:
* 6,281 samples
* 18 predictive features
* Target: future 30-day realized volatility
Train/Test Split:
* Training: 5,024
* Testing: 1,257
## Model Performance
| Model         | RMSE       | R²        |
| ------------- | ---------- | --------- |
| Random Forest | 0.1401     | 0.235     |
| XGBoost       | 0.1395     | 0.241     |
| LightGBM      | **0.1376** | **0.262** |
| LSTM          | 0.1507     | 0.111     |
| GRU           | 0.1398     | 0.235     |
<img width="1790" height="489" alt="image" src="https://github.com/user-attachments/assets/be0fc73e-bd89-4491-9800-7903c8ab3068" />

Top Predictive Features:
* 30-day realized volatility
* Jump intensity
* Momentum indicators

## Ensemble Model (Best Performance)
Combination: XGBoost + LSTM
* RMSE: **0.1352**
* R²: **0.2844**
* Performance improvement: **3.09%**
Insight:
> Hybrid models capture both nonlinear structure and temporal dynamics more effectively.

#  Options Pricing & Derivatives Analytics
Parameters:
* Spot: $87.23
* Strike: $95.95
* Maturity: 30 days
* Risk-free rate: 5%
* Volatility: 144.39%

## Black-Scholes Results
Call Price: **$11.1782**
Put Price: **$19.5076**

Greeks:
* Delta: 0.495
* Gamma: 0.011
* Theta: −0.244
* Vega: 0.0998
* Rho: 0.0263

## Monte Carlo Simulation
Call Price:
> **$6.6516**
95% CI:
> [$6.26, $7.04]
<img width="863" height="547" alt="image" src="https://github.com/user-attachments/assets/9365b6ad-fef6-4a9c-b284-e10ad72eddae" />

## Heston Stochastic Volatility Model
Call Price:
> **$6.0877**
Difference vs Black-Scholes:
> $5.09
Insight:
> Black-Scholes significantly overprices under extreme volatility regimes.

# 📊 Volatility Index Construction
A synthetic **Silver Volatility Index (SVIX)** was developed.
* Current value: **1239.73**
* Implied volatility proxy: **138.75%**
The volatility surface exhibits a clear **volatility smile**, consistent with commodity options markets.
<img width="761" height="658" alt="image" src="https://github.com/user-attachments/assets/ad865a6a-7739-45b1-8ecc-858da0a9e47e" />

#  Risk Management Framework
Portfolio Size: $1,000,000
## Value at Risk (1-Day)
| Confidence | Historical | Parametric |
| ---------- | ---------- | ---------- |
| 95%        | $31,873    | $33,858    |
| 99%        | $60,626    | $48,072    |

## Expected Shortfall
* ES 95%: $51,932
* ES 99%: $93,227
* ES/VaR Ratio: 1.54
Insight:
> Tail risk is significantly larger than Gaussian assumptions suggest.
<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/5da2684f-6902-475d-bead-2cd630ba462b" />

# Stress Testing
### Market Crash Scenario
* Price shock: −15%
* PnL: −$13,085
### Volatility Spike
* Vol increase: +30%
* New 99% VaR: $63,078

#  Trading Strategy Development
Initial Capital: $100,000
Backtest Window: Last 30 days
| Strategy            | Return     | Final Equity |
| ------------------- | ---------- | ------------ |
| Dual Moving Average | 1.28%      | $101,283     |
| RSI Mean Reversion  | 0.00%      | $100,000     |
| Breakout Pullback   | **15.93%** | **$115,925** |

Best Strategy:
> Breakout Pullback Momentum Strategy
Next-Day Signal:
> **NEUTRAL — No Trade**
<img width="1483" height="985" alt="image" src="https://github.com/user-attachments/assets/209880d4-dd68-418c-9a11-56adaadf7898" />

# Portfolio Optimization
Assets:
* Silver
* Gold
* S&P 500
Equal Weight Portfolio Volatility:
> 17.76% annually
Minimum Variance Portfolio:
> 17.76%
Insight:
> Correlation structure limited diversification benefits in this configuration.

# 📐 Volatility Term Structure
Using GARCH forecasts:
| Horizon  | Volatility |
| -------- | ---------- |
| 7 Days   | 0.28%      |
| 30 Days  | 0.28%      |
| 365 Days | 0.28%      |

# System Architecture Highlights
This project integrates multiple quantitative layers:
* Data Engineering Pipeline
* Statistical Modeling Engine
* Machine Learning Forecasting System
* Options Pricing Module
* Risk Analytics Engine
* Trading Strategy Backtester
* Portfolio Optimization Module
* Auto-Update Market Data System

# Key Insights & Takeaways
* Silver exhibits extreme kurtosis (>22), validating stochastic volatility models.
* Macro variables, especially gold and dollar index, significantly influence returns.
* Machine learning models outperform classical econometrics.
* Ensemble models provide the strongest predictive power.
* Stochastic volatility pricing is more realistic than Black-Scholes under stress.
* Momentum breakout strategies showed strong short-term performance.

# Future Enhancements
* Regime-switching volatility models
* Transformer-based forecasting
* Reinforcement learning trading agents
* Real options market calibration
* Live trading deployment

#  Author
**Ishu Verma**
Quantitative Finance
📧 [ishuverma1511@gmail.com](mailto:ishuverma1511@gmail.com)
📅 23 February 2026
