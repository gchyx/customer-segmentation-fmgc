import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os

pd.options.display.float_format = '{:.2f}'.format

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
transaction = pd.read_csv('Data_Analyst_Intern_Prework_Dataset.csv')

print("Data Types")
print(transaction.dtypes)

missing_info = pd.DataFrame({
    'Data Type': transaction.dtypes,
    'Missing Values': transaction.isnull().sum()
})
print("\nMissing Values")
print(missing_info)


# ─────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────

# Standardise customer names: strip whitespace, remove special characters and digits
transaction['Customer Name'] = (
    transaction['Customer Name']
    .str.strip()
    .str.replace(r'[^\w\s/]', '', regex=True)
    .str.replace(r'\d+', '', regex=True)
)

# Standardise brand to uppercase
transaction['Brand'] = transaction['Brand'].str.upper()

# Parse transaction dates (dayfirst=True handles DD/MM/YYYY format)
transaction['Transaction Date'] = pd.to_datetime(
    transaction['Transaction Date'], errors='coerce', dayfirst=True
)

print("\nCleaned Data Sample")
print(transaction.head())

# Save cleaned data
os.makedirs('output', exist_ok=True)
transaction.to_csv('output/transaction_cleaned.csv', index=False)


# ─────────────────────────────────────────────
# MARKET BASKET ANALYSIS
#  Purpose: Identify product categories frequently
#  purchased together to uncover cross-sell patterns
#  and validate Sugary Drinks purchase behaviour.
# ─────────────────────────────────────────────

# Create a basket matrix: rows = invoices, columns = categories, values = True/False
basket = transaction.groupby(['Invoice ID', 'Category']).size().unstack(fill_value=0)
basket = basket > 0

# Mine frequent itemsets with minimum 20% support
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

# Generate association rules filtered by lift > 1.0 (positive association)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0, num_itemsets=10)

print("\nAssociation Rules (Top Results)")
print(rules.head())

# Export rules
rules_csv = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
rules_csv.to_csv('output/association_rules.csv', index=False)


# ─────────────────────────────────────────────
# RFM SCORING
#  Scope: customers who purchased Sugary Drinks
#  and/or Culinary items only — most likely buyers
# f or the incoming Coca-Cola brand launch.
# ─────────────────────────────────────────────

transaction_filtered = transaction[
    transaction['Category'].isin(['SUGARY DRINKS', 'CULINARY'])
].copy()

# Recency: days since last purchase (lower = more recent = higher score)
transaction_recency = transaction_filtered.groupby(
    by='Customer Id', as_index=False
)['Transaction Date'].max()
transaction_recency.columns = ['Customer Id', 'LastPurchaseDate']

recent_date = transaction_recency['LastPurchaseDate'].max()
transaction_recency['Recency'] = transaction_recency['LastPurchaseDate'].apply(
    lambda x: (recent_date - x).days
)

# Frequency: total number of transactions per customer
transaction_freq = (
    transaction_filtered
    .drop_duplicates()
    .groupby(by='Customer Id', as_index=False)['Transaction Date']
    .count()
)
transaction_freq.columns = ['Customer Id', 'Frequency']

# Monetary: total spend per customer
transaction_filtered['Total'] = transaction_filtered['Sales Amount'] * transaction_filtered['Quantity']
transaction_mon = transaction_filtered.groupby(
    by='Customer Id', as_index=False
)['Total'].sum()
transaction_mon.columns = ['Customer Id', 'Monetary']

# Merge RFM components
transaction_rfm = (
    transaction_recency
    .merge(transaction_freq, on='Customer Id')
    .merge(transaction_mon, on='Customer Id')
    .drop(columns='LastPurchaseDate')
)

# Rank each component (normalised to 0–100)
transaction_rfm['R_rank'] = transaction_rfm['Recency'].rank(ascending=False)
transaction_rfm['F_rank'] = transaction_rfm['Frequency'].rank(ascending=True)
transaction_rfm['M_rank'] = transaction_rfm['Monetary'].rank(ascending=True)

transaction_rfm['R_rank_norm'] = (transaction_rfm['R_rank'] / transaction_rfm['R_rank'].max()) * 100
transaction_rfm['F_rank_norm'] = (transaction_rfm['F_rank'] / transaction_rfm['F_rank'].max()) * 100
transaction_rfm['M_rank_norm'] = (transaction_rfm['M_rank'] / transaction_rfm['M_rank'].max()) * 100  # fixed: was using F_rank

transaction_rfm.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

# Compute weighted RFM score (scaled to 1-5)
# Weights: Recency 15%, Frequency 28%, Monetary 57%
transaction_rfm['RFM_Score'] = (
    0.15 * transaction_rfm['R_rank_norm'] +
    0.28 * transaction_rfm['F_rank_norm'] +
    0.57 * transaction_rfm['M_rank_norm']
)
transaction_rfm['RFM_Score'] *= 0.05
transaction_rfm = transaction_rfm.round(2)
transaction_rfm.drop(columns=['R_rank_norm', 'F_rank_norm', 'M_rank_norm'], inplace=True)


# ─────────────────────────────────────────────
# CUSTOMER SEGMENTATION
# ─────────────────────────────────────────────

def categorize_score(score):
    if score >= 4.5:
        return 'Champions'           # Frequent, recent, high spenders
    elif score >= 3.5:
        return 'Potential Loyalists' # Fairly frequent, strong potential
    elif score >= 2.5:
        return 'New Customers'       # Recent but low frequency
    elif score >= 1.5:
        return 'At Risk Customers'   # Declining engagement
    else:
        return 'Lost Customers'      # Rarely purchase

transaction_rfm['Customer Category'] = transaction_rfm['RFM_Score'].apply(categorize_score)

# Attach customer names
transaction_rfm = transaction_rfm.merge(
    transaction_filtered[['Customer Id', 'Customer Name']].drop_duplicates(),
    on='Customer Id',
    how='left'
)

# Export top 100 customers by RFM score
top_100 = transaction_rfm.sort_values('RFM_Score', ascending=False).head(100)
print("\nTop 100 Customers")
print(top_100)

top_100.to_csv('output/top_customers.csv', index=False)
print("\nAnalysis complete. Results saved to /output.")
