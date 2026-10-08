import pandas as pd, numpy as np
df = pd.read_csv(r'D:\Internproj\New ML\clean_engineered_dataset.csv', parse_dates=['Date'])
df = df.sort_values(['Route','Date'])
df['log_price'] = np.log(df['Price_USD'].clip(lower=1))
df['ret'] = df.groupby('Route')['log_price'].diff()
df['sq'] = df['ret']**2
for H in [1,5,20]:
    df['fr%d' % H] = df.groupby('Route')['ret'].transform(
        lambda s, H=H: s.shift(-1).rolling(H, min_periods=H).sum().shift(-(H-1)))
df['rv22'] = df.groupby('Route')['sq'].transform(
    lambda s: np.sqrt(s.rolling(22, min_periods=22).mean()) * np.sqrt(252))
print('--- price describe by route ---')
print(df.groupby('Route')['Price_USD'].describe()[['min','25%','50%','75%','max']].to_string())
print('\n--- daily ret std by route ---')
print(df.groupby('Route')['ret'].std().to_string())
print('\n--- rv22 median by route ---')
print(df.groupby('Route')['rv22'].median().to_string())
print('\n--- targets ---')
for H in [1,5,20]:
    v = df['fr%d' % H].dropna()
    pp = (v > 0).mean()
    print('H=%d n=%d frac_pos=%.4f maj=%.4f sd=%.5f  mean|r|=%.5f' % (H, len(v), pp, max(pp, 1-pp), v.std(), v.abs().mean()))
    for t in [0.02, 0.03, 0.05]:
        print('    fixed %.0f%%: rate=%.4f' % (t*100, (v.abs() > t).mean()))
    m = df['fr%d' % H].notna() & df['rv22'].notna()
    thr = df.loc[m, 'rv22'] * np.sqrt(H/252.0)
    for k in [1.0, 1.5, 2.0]:
        print('    vol k=%.1f: rate=%.4f' % (k, (df.loc[m, 'fr%d' % H].abs() > k*thr).mean()))
print('\n--- Is_Peak_Season ---')
print(df.groupby('Is_Peak_Season').size().to_string())
print(df.groupby(df['Date'].dt.month)['ret'].std().to_string())
print('\n--- warmup NaN counts for rv_trail_66, rv_pct_rank ---')
print('total rows', len(df))
