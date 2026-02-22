#  Forecasting Silver Market Volatility: A Comparative Study of Econometric and Machine Learning Models with Applications to Option Pricing and Risk Management
A full-stack quantitative research project analyzing **Silver Futures** volatility using statistical models, machine learning, and derivative pricing techniques.

This project covers:
* Long-horizon financial data engineering (25+ years)
* Volatility modeling (EWMA, GARCH family)
* Machine learning forecasting (RF, XGBoost, LightGBM, LSTM, GRU)
* Options pricing (Black-Scholes, Monte Carlo, Heston)
* Risk management (VaR, Expected Shortfall, Stress Testing)
* Trading strategy backtesting
* Regime detection & crash prediction
* Portfolio optimization & volatility term structure
  <img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/8f9e6fb6-c2aa-48d4-a411-d3ce8830557c" />

# 📊 1. Data Acquisition & Preprocessing
## Silver Futures Data
* Source: Yahoo Finance
* Observations downloaded: **6,394**
* Time range: **2000-08-30 → 2026-02-20**
After cleaning and return calculation:
* Final dataset size: **6,393 observations**
* Final range: **2000-08-31 → 2026-02-20**

## Macro Variables Included
To capture cross-asset volatility transmission:
* Gold
* Dollar Index (DXY)
* US 10-Year Treasury Yield
* VIX
* S&P 500
These were aligned temporally with the silver dataset.

# 📈 2. Returns Distribution Analysis
### Silver Return Statistics
| Metric             | Value         |
| ------------------ | ------------- |
| Mean               | **0.000440**  |
| Standard Deviation | **0.020847**  |
| Skewness           | **−1.551612** |
| Kurtosis           | **22.478674** |
| Minimum            | **−0.376103** |
| Maximum            | **0.131250**  |

Key insight:
* Strong **negative skew**
* Extremely **fat tails** (kurtosis > 22)
* Presence of crash risk and jumps

## Normality Test
Jarque-Bera Test:
* p-value = **0.0000000000**
Conclusion:
> Returns are **not normally distributed**, validating the need for GARCH-type and jump-aware models.

# ⚡ 3. Volatility Feature Engineering
## Realized Volatility
Annualized realized volatility windows:
| Feature | Mean         |
| ------- | ------------ |
| rv_7    | **0.278447** |
| rv_30   | **0.292772** |
| rv_90   | **0.297279** |

Observation:
* Volatility clustering is persistent across horizons.

## Jump Detection
* Total jumps detected: **324**
* Jump frequency: **5.07%**
This confirms silver behaves as a **jump-diffusion asset** rather than pure diffusion.

## Macro Correlations
Correlation with silver returns:
| Asset        | Correlation   |
| ------------ | ------------- |
| Gold         | **0.777315**  |
| Dollar Index | **−0.356694** |

Insights:
* Silver behaves strongly like a **leveraged gold proxy**
* USD strength negatively impacts silver

# 📉 4. Volatility Modeling (Statistical Models)
## EWMA Model
* Current EWMA volatility: **1.2990**

## GARCH(1,1)
5-day volatility forecast (Feb 20, 2026):
| Horizon | Forecast     |
| ------- | ------------ |
| h.1     | **1.256609** |
| h.5     | **1.251405** |

## EGARCH
* Asymmetry parameter (gamma): **−0.0315459**
Interpretation:
> Negative shocks increase volatility more than positive shocks (leverage effect).
## Model Performance (RMSE vs Realized Vol)
| Model  | RMSE       |
| ------ | ---------- |
| EWMA   | **0.0334** |
| GARCH  | **0.0375** |
| EGARCH | **0.0497** |

EWMA performed best among classical models.

# 🤖 5. Machine Learning Volatility Forecasting
## Dataset
* Features: **18**
* Samples: **6,280**
* Train size: **5,024**
* Test size: **1,256**
* Train/Test split date: **2021-01-05**

## Random Forest
* RMSE: **0.1378**
* R²: **0.2300**
Top features:
* rv_30
* jump_intensity_30
* MACD

## XGBoost
* RMSE: **0.1372**
* R²: **0.2372**
Top features:
* jump_intensity_30
* rv_30
* MACD

## LightGBM (Best Model)
* RMSE: **0.1355**
* R²: **0.2552**
  
## Deep Learning Models
### LSTM
* RMSE: **0.1513**
* R²: **0.0673**

### GRU
* RMSE: **0.1441**
* R²: **0.1542**
Observation:
> Tree-based models outperform deep learning for tabular financial data.

## Ensemble Model (XGBoost + LSTM)
* RMSE: **0.1360**
* R²: **0.2464**
Improvement over XGBoost:
* **0.84% RMSE reduction**
  
# 💰 6. Options Pricing & Greeks
## Market Assumptions
* Silver Price: **$82.34**
* Strike: **$90.58** (10% OTM)
* Maturity: **30 days**
* Risk-free rate: **5%**
* Volatility: **143.89%**

## Black-Scholes Pricing
Call Price: **$10.5051**
Put Price: **$18.3679**

## Greeks (Call)

| Greek | Value         |
| ----- | ------------- |
| Delta | **0.494090**  |
| Gamma | **0.011743**  |
| Theta | **−0.229968** |
| Vega  | **0.094168**  |
| Rho   | **0.024805**  |

## Implied Volatility
Market call price assumption: **$2.50**
* Implied Volatility: **97.47%**
* IV − HV Spread: **−46.42%**
Indicates potential **volatility overestimation** in historical measures.

## Monte Carlo Pricing
* Call price: **$6.2358**
Difference from Black-Scholes:
* **$4.2693**

## Heston Model Pricing
* Call price: **$5.7055**
Difference from Black-Scholes:
* **$4.7996**
Conclusion:
> Black-Scholes significantly overprices options when stochastic volatility is considered.

# 🛡️ 7. Risk Management
## Silver Volatility Index (SVIX)
* Current value: **1240.93**
* Implied volatility proxy: **138.88%**

## Value at Risk (VaR)
Position size: **$1,000,000**

### 95% Confidence
* Historical VaR: **$31,874.20**
* Parametric VaR: **$33,849.34**
* 
### 99% Confidence
* Historical VaR: **$60,626.15**
* Parametric VaR: **$48,056.23**

## Expected Shortfall (ES)
### 95% ES
* **$51,932.46**
### 99% ES
* **$93,227.44**
Tail Risk Ratio:
* ES₉₉ / VaR₉₉ = **1.54**
Indicates heavy tail exposure.

## Stress Testing
| Scenario                  | P&L                   |
| ------------------------- | --------------------- |
| Market Crash (−15%)       | **−$12,351**          |
| Silver Flash Crash (−25%) | **−$20,586**          |
| Volatility Spike (+30%)   | New VaR ≈ **$63,046** |

# 📊 8. Trading Strategy Analysis
## Volatility Arbitrage Strategy
Initial Capital: **$100,000**
Final Equity: **$5,179**
Performance:
| Metric        | Value        |
| ------------- | ------------ |
| Total Return  | **−94.82%**  |
| Buy & Hold    | **300.39%**  |
| Excess Return | **−395.22%** |
| Sharpe Ratio  | **−0.449**   |
| Max Drawdown  | **−95.55%**  |

Insight:
> Naive volatility arbitrage without regime filtering is unprofitable.

## Regime-Based Strategy (HMM — 3 Regimes)
| Regime | Mean Return | Volatility | Sharpe |
| ------ | ----------- | ---------- | ------ |
| 0      | **12.18%**  | 17.45%     | 0.67   |
| 1      | **−21.59%** | 49.79%     | −0.39  |
| 2      | **25.51%**  | 28.08%     | 0.91   |

Regime modeling significantly improves risk understanding.

## Crash Prediction Model
* Current crash probability: **0.30%**

# 📊 9. Portfolio Analysis
Assets:
* Silver
* Gold
* S&P 500

## Portfolio Volatility
Equal Weight:
* **17.75%**
Minimum Variance:
* **17.75%**
(No reduction due to correlation structure in this sample)

## Volatility Term Structure
30-day expected volatility:
* **34.58%**
* 
# 🧠 Key Insights
1. Silver exhibits **extreme fat tails and jump risk**
2. Macro variables (especially gold and USD) strongly influence returns
3. Classical EWMA performed surprisingly well vs GARCH
4. Machine learning improves forecasting modestly
5. Black-Scholes overprices under stochastic volatility assumptions
6. Tail risk is significant (ES/VaR = 1.54)
7. Regime detection is critical for profitable trading

# 🚀 Technologies Used
* Python
* NumPy / Pandas
* Scikit-Learn
* XGBoost / LightGBM
* TensorFlow / Keras
* Statsmodels / ARCH
* Matplotlib / Seaborn
* yFinance

# 👤 Ishu Verma
Quantitative Research Project demonstrating:
* Derivatives knowledge
* Machine learning in finance
* Risk modeling
* Systematic trading development

If you found this project useful, feel free to ⭐ the repo.
Contact - ishuverma1511@gmail.com 
