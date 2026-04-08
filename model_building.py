# Churn patterns + key drivers + lifetime/cohort trends on cust_df (synthetic demo dataset)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

# Basic time fields
cust_df['acq_month'] = cust_df['acquisition_date'].dt.to_period('M').dt.to_timestamp()
cust_df['churn_month'] = cust_df['churn_date'].dt.to_period('M').dt.to_timestamp()

# Helper aggregations
monthly_acq = cust_df.groupby('acq_month', as_index=False).agg(new_customers=('customer_id','count'))
monthly_churn = cust_df[cust_df['churned']==1].groupby('churn_month', as_index=False).agg(churned_customers=('customer_id','count'))

# Active base over time (approx): acquired to date minus churned to date
calendar_months = pd.date_range(cust_df['acq_month'].min(), pd.Timestamp('2026-03-01'), freq='MS')
cal_df = pd.DataFrame({'month': calendar_months})
cal_df = cal_df.merge(monthly_acq.rename(columns={'acq_month':'month'}), on='month', how='left')
cal_df = cal_df.merge(monthly_churn.rename(columns={'churn_month':'month'}), on='month', how='left')
cal_df[['new_customers','churned_customers']] = cal_df[['new_customers','churned_customers']].fillna(0)
cal_df['cum_acq'] = cal_df['new_customers'].cumsum()
cal_df['cum_churn'] = cal_df['churned_customers'].cumsum()
cal_df['active_base_est'] = (cal_df['cum_acq'] - cal_df['cum_churn']).clip(lower=0)
cal_df['logo_churn_rate_est'] = np.where(cal_df['active_base_est'].shift(1).fillna(cal_df['active_base_est'])>0,
                                         cal_df['churned_customers'] / cal_df['active_base_est'].shift(1).fillna(cal_df['active_base_est']),
                                         np.nan)

# Plot: acquisitions, churns, active base
plt.figure(figsize=(12,5))
plt.plot(cal_df['month'], cal_df['active_base_est'], label='Active base (est)')
plt.title('Active customer base over time (estimated)')
plt.xlabel('Month')
plt.ylabel('Customers')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(cal_df['month'], cal_df['new_customers'], label='New customers')
plt.plot(cal_df['month'], cal_df['churned_customers'], label='Churned customers')
plt.title('New vs churned customers by month')
plt.xlabel('Month')
plt.ylabel('Customers')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(cal_df['month'], cal_df['logo_churn_rate_est']*100)
plt.title('Estimated monthly logo churn rate')
plt.xlabel('Month')
plt.ylabel('Churn rate (%)')
plt.tight_layout()
plt.show()

# Tenure distribution (churned only) + overall
plt.figure(figsize=(10,4))
sns.histplot(cust_df.loc[cust_df['churned']==1, 'tenure_months'], bins=30, kde=False)
plt.title('Tenure at churn (months)')
plt.xlabel('Tenure months')
plt.ylabel('Customers')
plt.tight_layout()
plt.show()

# Driver slices: churn rate by key dimensions
slice_cols = ['billing_cycle','plan','segment','channel','activated_30d','payment_failure_flag']
churn_slices = []
for col in slice_cols:
    tmp = cust_df.groupby(col, as_index=False).agg(customers=('customer_id','count'), churn_rate=('churned','mean'),
                                                  avg_tenure=('tenure_months','mean'), avg_ltv=('lifetime_revenue','mean'))
    tmp['dimension'] = col
    tmp = tmp.rename(columns={col:'value'})
    churn_slices.append(tmp)
churn_slices_df = pd.concat(churn_slices, ignore_index=True)

# Show top differences by churn rate within each dimension
summary_tables = {}
for dim in churn_slices_df['dimension'].unique():
    df_dim = churn_slices_df[churn_slices_df['dimension']==dim].sort_values('churn_rate', ascending=False)
    summary_tables[dim] = df_dim

for dim, df_dim in summary_tables.items():
    print('\
' + dim)
    print(df_dim.to_string(index=False))

# Cohort retention: by acquisition month and age (month index)
# Compute age in months at churn/censor as of end_date
obs_end_date = pd.Timestamp('2026-03-31')
obs_end = pd.Series(np.where(cust_df['churned']==1, cust_df['churn_date'].values.astype('datetime64[ns]'), obs_end_date.to_datetime64())).astype('datetime64[ns]')
age_months = np.maximum(0, np.floor(((obs_end - cust_df['acquisition_date']).dt.days)/30).astype(int))
# For retention, we want for each cohort and age k: retained if age_months >= k
cust_df['age_months_obs'] = age_months

# Keep cohorts with decent size
cohort_sizes = cust_df.groupby('acq_month').size().rename('cohort_size')
valid_cohorts = cohort_sizes[cohort_sizes>=300].index

max_age = 24
cohort_rows = []
for cohort in valid_cohorts:
    sub = cust_df[cust_df['acq_month']==cohort]
    size = len(sub)
    for k in range(0, max_age+1):
        retained = (sub['age_months_obs'] >= k).mean()
        cohort_rows.append({'cohort_month': cohort, 'age_months': k, 'retention': retained, 'cohort_size': size})
cohort_df = pd.DataFrame(cohort_rows)

# Pivot heatmap
cohort_piv = cohort_df.pivot(index='cohort_month', columns='age_months', values='retention').sort_index()
plt.figure(figsize=(12,7))
sns.heatmap(cohort_piv, cmap='Blues', vmin=0, vmax=1)
plt.title('Cohort retention heatmap (share retained)')
plt.xlabel('Age (months)')
plt.ylabel('Acquisition cohort (month)')
plt.tight_layout()
plt.show()

# LTV / CLV trends: by cohort month
ltv_by_cohort = cust_df.groupby('acq_month', as_index=False).agg(
    customers=('customer_id','count'),
    avg_ltv=('lifetime_revenue','mean'),
    median_ltv=('lifetime_revenue','median'),
    churn_rate=('churned','mean'),
    annual_share=('billing_cycle', lambda x: (x=='Annual').mean()),
    avg_engagement=('engagement_30d','mean'),
    activation_rate=('activated_30d','mean')
)

plt.figure(figsize=(12,4))
plt.plot(ltv_by_cohort['acq_month'], ltv_by_cohort['avg_ltv'])
plt.title('Average lifetime revenue by acquisition cohort')
plt.xlabel('Cohort month')
plt.ylabel('Avg lifetime revenue')
plt.tight_layout()
plt.show()

# Model drivers quickly: logistic regression for churn probability
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

feature_cols = ['segment','channel','plan','billing_cycle','mrr','activated_30d','engagement_30d','support_tickets_90d','payment_failure_flag']
X = cust_df[feature_cols].copy()
y = cust_df['churned'].astype(int)

cat_cols = ['segment','channel','plan','billing_cycle']
num_cols = [c for c in feature_cols if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

clf = Pipeline(steps=[
    ('prep', preprocess),
    ('lr', LogisticRegression(max_iter=200, n_jobs=None))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
clf.fit(X_train, y_train)

pred = clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, pred)
print('\
AUC:')
print(auc)

# Extract coefficients
ohe = clf.named_steps['prep'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(cat_cols)
feature_names = list(cat_feature_names) + num_cols
coefs = clf.named_steps['lr'].coef_[0]
coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
coef_df['odds_mult_per_unit'] = np.exp(coef_df['coef'])
coef_df_sorted = coef_df.reindex(coef_df['coef'].abs().sort_values(ascending=False).index).head(20)
print('\
Top driver coefficients (log-odds):')
print(coef_df_sorted.to_string(index=False))

# Visualize top coefficients
plot_df = coef_df_sorted.sort_values('coef')
plt.figure(figsize=(10,6))
plt.barh(plot_df['feature'], plot_df['coef'])
plt.title('Top churn drivers (logistic regression coefficients)')
plt.xlabel('Coefficient (log-odds)')
plt.tight_layout()
plt.show()
