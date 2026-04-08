# Create a synthetic subscription churn dataset so we can demonstrate end-to-end churn pattern + CLV/cohort analysis
# (No customer dataset is present in the workspace.)

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

n_customers = 25000
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2026-03-31')

def random_dates(n, start, end, rng_obj):
    delta_days = (end - start).days
    offsets = rng_obj.integers(0, delta_days + 1, size=n)
    return start + pd.to_timedelta(offsets, unit='D')

customer_id = np.arange(1, n_customers + 1)
acq_date = random_dates(n_customers, start_date, end_date - pd.Timedelta(days=30), rng)

# Segment + plan + channel
segments = rng.choice(['SMB', 'MidMarket', 'Enterprise'], size=n_customers, p=[0.72, 0.22, 0.06])
channels = rng.choice(['Paid Search', 'Paid Social', 'Organic', 'Partner', 'Outbound'], size=n_customers, p=[0.22, 0.17, 0.32, 0.12, 0.17])
plan = rng.choice(['Basic', 'Pro', 'Business'], size=n_customers, p=[0.55, 0.33, 0.12])
billing = rng.choice(['Monthly', 'Annual'], size=n_customers, p=[0.82, 0.18])

base_price = pd.Series(plan).map({'Basic': 29, 'Pro': 59, 'Business': 129}).to_numpy(dtype=float)
# annual discount
mrr = np.where(billing == 'Annual', base_price * 0.85, base_price)

# Simulate first-30-day activation and early engagement score
activated_30d = rng.binomial(1, p=np.where(plan=='Basic', 0.62, np.where(plan=='Pro', 0.70, 0.76)))
engagement_30d = np.clip(rng.normal(loc=0.55 + 0.12*activated_30d, scale=0.18, size=n_customers), 0, 1)

# Support tickets in first 90d
support_tickets_90d = rng.poisson(lam=np.where(segments=='Enterprise', 1.3, 0.9) + (1-engagement_30d)*1.2)

# Payment failure propensity
payment_fail_flag = rng.binomial(1, p=np.where(billing=='Monthly', 0.10, 0.04) + (channels=='Paid Social')*0.02)

# Hazard model for churn: baseline depends on month age; add multipliers for drivers
# We'll draw churn month for each customer; if none, they're active at end_date.
max_months = 40

# baseline hazards by age in months (higher early churn, stabilizes, slight re-acceleration later)
age = np.arange(1, max_months+1)
baseline_h = 0.06*np.exp(-age/7) + 0.012 + 0.004*(age>24)

# customer-specific risk score
risk = (
    -0.55*activated_30d
    -0.85*engagement_30d
    +0.22*(support_tickets_90d>=3).astype(int)
    +0.35*(support_tickets_90d>=6).astype(int)
    +0.55*payment_fail_flag
    +0.18*(plan=='Basic').astype(int)
    -0.10*(plan=='Business').astype(int)
    -0.25*(billing=='Annual').astype(int)
    -0.20*(segments=='Enterprise').astype(int)
    +0.08*(channels=='Paid Social').astype(int)
)

# Convert to multiplier with bounded effect
mult = np.exp(np.clip(risk, -1.2, 1.4))

# Generate churn month via sequential survival
churn_month = np.full(n_customers, fill_value=np.nan)
for m in range(1, max_months+1):
    # probability of churn in this month given survival so far
    h_m = baseline_h[m-1] * mult
    h_m = np.clip(h_m, 0, 0.35)
    alive = np.isnan(churn_month)
    churn_now = (rng.random(n_customers) < h_m) & alive
    churn_month[churn_now] = m

# Determine churn date and status
# monthly customers churn at acquisition_date + churn_month months (approx); annual churn at anniversary month
acq_month_start = acq_date.values.astype('datetime64[M]').astype('datetime64[ns]')

# Create churn_date
churn_date = pd.to_datetime(pd.NaT)
churn_date_arr = np.array([np.datetime64('NaT')] * n_customers, dtype='datetime64[ns]')

for idx in range(n_customers):
    if np.isnan(churn_month[idx]):
        continue
    m = int(churn_month[idx])
    if billing[idx] == 'Monthly':
        churn_date_arr[idx] = (pd.Timestamp(acq_date[idx]) + pd.DateOffset(months=m)).to_datetime64()
    else:
        # annual: churn only at renewal points; approximate churn month snapped to nearest 12 multiple
        snap = int(np.ceil(m/12.0)*12)
        churn_date_arr[idx] = (pd.Timestamp(acq_date[idx]) + pd.DateOffset(months=snap)).to_datetime64()

churn_date = pd.to_datetime(churn_date_arr)

# censor at end_date
churned = (~churn_date.isna()) & (churn_date <= end_date)
churn_date = churn_date.where(churned, pd.NaT)

# Tenure months (capped)
end_obs = pd.Series(np.where(churned, churn_date.values.astype('datetime64[ns]'), end_date.to_datetime64())).astype('datetime64[ns]')
tenure_days = (end_obs - acq_date).dt.days
# approximate months
tenure_months = np.maximum(1, np.floor(tenure_days/30).astype(int))

# Lifetime revenue approximation
# Monthly: MRR * tenure_months; Annual: upfront annual billed each 12 months, use 12*mrr as annualized
lifetime_revenue = np.where(
    billing == 'Monthly',
    mrr * tenure_months,
    (12*mrr) * np.maximum(1, np.floor(tenure_months/12).astype(int))
)

# Build dataframe
cust_df = pd.DataFrame({
    'customer_id': customer_id,
    'acquisition_date': acq_date,
    'segment': segments,
    'channel': channels,
    'plan': plan,
    'billing_cycle': billing,
    'mrr': mrr,
    'activated_30d': activated_30d,
    'engagement_30d': engagement_30d,
    'support_tickets_90d': support_tickets_90d,
    'payment_failure_flag': payment_fail_flag,
    'churned': churned.astype(int),
    'churn_date': churn_date,
    'tenure_months': tenure_months,
    'lifetime_revenue': lifetime_revenue
})

print(cust_df.head(10).to_string(index=False))
print('\
Rows:')
print(len(cust_df))
print('\
Churn rate (observed):')
print(cust_df['churned'].mean())
