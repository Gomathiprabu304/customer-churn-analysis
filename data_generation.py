# Load likely customer/subscription churn dataset(s) from the working directory and do a first-pass EDA
import os
import pandas as pd

files = sorted(os.listdir('.'))
print(pd.Series(files).head(80).to_string(index=False))

# Try to find common data file types
candidates = [f for f in files if f.lower().endswith(('.csv', '.parquet', '.xlsx', '.xls', '.json'))]
print('\
')
print(pd.Series(candidates).to_string(index=False))

# Heuristic pick: prefer files with churn/subscription/customer in name
key_terms = ['churn','subscription','subs','customer','users','cohort','retention','billing','payments','mrr','arr']
ranked = []
for f in candidates:
    score = sum([1 for t in key_terms if t in f.lower()])
    ranked.append((score, f))
ranked_sorted = sorted(ranked, reverse=True)
print('\
Top candidates:')
print(pd.DataFrame(ranked_sorted, columns=['score','file']).head(15).to_string(index=False))

chosen_file = ranked_sorted[0][1] if ranked_sorted else None
print('\
Chosen file:')
print(chosen_file)

# Load into df if possible
if chosen_file is None:
    df = None
else:
    low = chosen_file.lower()
    if low.endswith('.csv'):
        df = pd.read_csv(chosen_file)
    elif low.endswith('.parquet'):
        df = pd.read_parquet(chosen_file)
    elif low.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(chosen_file)
    elif low.endswith('.json'):
        df = pd.read_json(chosen_file)
    else:
        df = None

if df is not None:
    print('\
Data shape:')
    print(df.shape)
    print('\
Head:')
    print(df.head(10).to_string(index=False))
else:
    print('\
No dataset loaded into df.')
