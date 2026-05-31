# Customer Targeting Analysis: RFM Segmentation & Market Basket Analysis

## Overview

A beverage supplier is launching the Coca-Cola brand and needs to acquire **100 customers within a 5-day window**. This project identifies the top 100 highest-potential customers from historical transaction data using two analytical techniques:

- **Market Basket Analysis**: to uncover which product categories are frequently bought alongside Sugary Drinks, expanding the pool of targetable customers
- **RFM Scoring**: to rank and segment customers by purchasing behaviour (Recency, Frequency, Monetary value)

The final output is a prioritised list of 100 customers with actionable segmentation to guide targeted outreach.

---

## Key Findings

- Customers who purchase **Culinary items** have a **73% likelihood** of also purchasing Sugary Drinks (lift: 2.51), significantly expanding the targetable customer base beyond direct Sugary Drinks buyers
- Of the top 100 customers: **16% are Champions**, **57% are Potential Loyalists**, and **27% are New Customers**
- Champions and Potential Loyalists should be prioritised for direct outreach; New Customers respond better to introductory or cross-promotional offers

---

## Methodology

### 1. Market Basket Analysis
Used the Apriori algorithm to mine frequent itemsets across product categories (minimum support: 20%). Association rules were filtered by lift > 1.0 to identify statistically meaningful co-purchase patterns.

### 2. RFM Scoring
Customers who purchased Sugary Drinks and/or Culinary items were scored across three dimensions:

| Dimension | Definition | Weight |
|-----------|-----------|--------|
| Recency | Days since last purchase | 15% |
| Frequency | Total number of transactions | 28% |
| Monetary | Total spend | 57% |

Each dimension was ranked and normalised (0–100), then combined into a weighted RFM score scaled to 1–5.

### 3. Customer Segmentation

| Segment | RFM Score | Description |
|---------|-----------|-------------|
| Champions | ≥ 4.5 | Frequent, recent, high spenders |
| Potential Loyalists | ≥ 3.5 | Fairly frequent with strong potential |
| New Customers | ≥ 2.5 | Recently started, low frequency |
| At Risk | ≥ 1.5 | Declining engagement |
| Lost Customers | < 1.5 | Minimal engagement |


---

## Tools & Libraries

- **Python** — pandas, mlxtend
- **Tableau** — Sales dashboard

---

## Deliverables

| File | Description |
|------|-------------|
| `case_study.pdf` | End-to-end case study deck |
| [Sales Dashboard](https://public.tableau.com/views/SalesDashboard_17523886777230/Dashboard?:language=en-US&:sid=&:display_count=n&:origin=viz_share_link) | Tableau sales performance dashboard |
