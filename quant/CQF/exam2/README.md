# Monte Carlo Option Pricing with Numerical Schemes

A comprehensive implementation of Monte Carlo simulation techniques for European and binary option pricing, comparing multiple numerical integration schemes and variance reduction strategies.

## Project Highlights

This project demonstrates proficiency in:
- **Quantitative Finance**: Risk-neutral valuation, option pricing theory, stochastic processes
- **Numerical Methods**: Euler-Maruyama and Milstein schemes for SDE simulation
- **Python Engineering**: NumPy vectorization, data analysis, scientific computing
- **Research & Analysis**: Convergence analysis, variance reduction, sensitivity studies

## Problem Overview

Implement the risk-neutral valuation formula to price derivatives:

$$V(S, t) = e^{-r(T-t)} \mathbb{E}^Q[\text{Payoff}]$$

For a stock following geometric Brownian motion, compare multiple numerical approaches to determine pricing accuracy and computational efficiency.

## Implementation

### Simulation Schemes
1. **Euler-Maruyama** - Standard first-order discretization
2. **Milstein Scheme** - Second-order improvement with reduced bias
3. **Analytical Solution** - Black-Scholes closed form for European options

### Option Payoffs
- **European Call**: `max(S_T - E, 0)`
- **Binary Call**: Various cash/asset-or-nothing structures

### Variance Reduction
- **Antithetic Variates**: Leverage negative correlation to reduce variance

## Analysis Dimensions

The notebook provides:
- **Convergence Analysis**: Error comparison across schemes as number of simulations increases
- **Parameter Sensitivity**: Impact of S₀, E, T, σ, r on option pricing
- **Variance Efficiency**: Quantitative comparison of variance reduction effectiveness
- **Computational Performance**: Time and accuracy trade-offs

## Key Parameters

Default calibration (can be varied for sensitivity analysis):
- Initial stock price: 100
- Strike: 100  
- Maturity: 1 year
- Volatility: 20% p.a.
- Risk-free rate: 5% p.a.

## Stack

- **Python 3.8+**
- **NumPy**: Vectorized simulations
- **Matplotlib**: Visualization
- **Jupyter Notebook**: Interactive analysis and reporting

## Structure

The notebook contains:
1. Theoretical foundation and setup
2. Algorithm implementations and validation
3. Empirical results with comprehensive tables
4. Visual analysis of convergence and sensitivity
5. Key observations and insights
6. Academic references

## Insights & Learnings

- Comparison of weak vs. strong convergence properties
- Trade-offs between computational cost and accuracy
- Importance of variance reduction in Monte Carlo methods
- Parameter sensitivities in derivatives pricing
