# Global Real Estate Dynamics: A Multi-Source Statistical Analysis (2015-2024)
> **Author:** Dimo Dimov  
> **Date:** April 2026  
> **Affiliation:** SoftUni Data Science  
> **Status:** OSF Preregistration Style Documentation

---

## 1. Project Abstract
This research presents a multi-dimensional statistical audit of the global housing market spanning a decade (2015-2024). By integrating high-fidelity datasets from Kaggle, Yahoo Finance, FRED, and the World Bank, the study explores the interplay between property valuations and macroeconomic drivers across 20 countries. The methodology transitions from baseline linear modeling ($R^2 \approx 0.06$) to an advanced Engineered Feature Framework ($R^2 \approx 0.53$).

## 2. Research Hypotheses (Preregistered)
*   **H1 (Supply-Side):** Structural inventory deficits (Construction Index) have a higher predictive weight on house prices than demand-side metrics (GDP Growth).
*   **H2 (Monetary Impact):** High-interest rate environments show a delayed negative correlation with affordability ratios.
*   **H3 (Financialization):** Global housing markets exhibit significant synchronization with equity market performance (S&P 500).

## 3. Data Inventory & Provenance
The analysis utilizes four distinct primary data streams:
1.  **Housing Core:** Global House Price Index, Rent Index, and Affordability ratios (Source: Kaggle/atharvasoundankar).
2.  **Equity & Commodities:** S&P 500 (^GSPC), Gold (GC=F), and Crude Oil (CL=F) (Source: Yahoo Finance API).
3.  **Monetary Policy:** Federal Funds Rate and M2 Money Supply (Source: FRED via `pandas_datareader`).
4.  **Demographics:** Total Population and Unemployment Rate (Source: World Bank API).

## 4. Analytical Workflow
### Phase I: Tidy Data & Normalization
*   Merging disparate APIs on standardized temporal (Year) and geographic (ISO-3) keys.
*   Application of **Min-Max Scaling** to ensure feature comparability across different magnitudes (e.g., Millions of people vs. Percentage rates).

### Phase II: Feature Engineering
*   **Supply Pressure:** Calculated as $\frac{\text{House Price Index}}{\text{Construction Index}}$.
*   **Real Interest Rate:** Calculated as $\text{Mortgage Rate} - \text{Inflation Rate}$.

### Phase III: Statistical Inference
*   **Forensic Auditing:** Normality audits (Shapiro-Wilk) and Benford’s Law application.
*   **Multivariate Modeling:** Ordinary Least Squares (OLS) Regression with Residual Analysis (Durbin-Watson).
*   **Collinearity Audits:** Variance Inflation Factor (VIF) to detect Information Overlap.

## 5. Key Statistical Discoveries
*   **Primary Pivot:** *Supply Pressure* identified as the dominant predictor ($P < 0.001$).
*   **Financial Impact:** Equity markets account for approximately 40.1% of relative price variance in specific clusters.
*   **The Paradox:** High interest rates are statistically secondary to structural supply deficits in driving global valuations.

## 6. Environment & Dependencies
*   **Language:** Python 3.x
*   **Libraries:** `pandas`, `numpy`, `yfinance`, `statsmodels`, `scikit-learn`, `seaborn`, `plotly`, `kagglehub`.

## 7. Media & Additional Resources

### Project Evolution Visualization (Gource)
A 20-second time-lapse representing the repository's structural growth and commit history.

![Project Timeline Visualization](./Additional%20resources/gource_project_visualization.gif)

### Statistical Framework & Integrated Insights
A comprehensive visual synthesis of the multi-source data integration and the finalized statistical model results.

![Statistical Synthesis](./Additional%20resources/README-merged-image.png)

---
*Created as part of the SoftUni Data Science Program (2026).*