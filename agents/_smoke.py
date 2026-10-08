import sys, time
sys.path.insert(0, r'D:\testing\agents')
from agentB_common import *
t0 = time.time()
raw = load_raw()
mir = verify_mirrors(raw)
print(mir.to_string(index=False))
d = build_features(raw)
d = lag_weather(d)
fs, p, w, cal = feature_sets(d)
print('price', len(p), 'weather', len(w), 'cal', len(cal))
print('PRICE:', p)
print()
print('WEATHER:', w)
print()
# leakage sanity: univariate AUC of each feature against fwd_ret_1 sign
from sklearn.metrics import roc_auc_score
y = (d['fwd_ret_1'] > 0).astype(int).to_numpy()
m = np.isfinite(d['fwd_ret_1'].to_numpy())
rows = []
for c in fs['price_weather']:
    x = d[c].astype(float).to_numpy()
    ok = m & np.isfinite(x)
    if ok.sum() < 1000 or len(np.unique(y[ok])) < 2:
        continue
    a = roc_auc_score(y[ok], x[ok])
    rows.append((c, max(a, 1-a)))
rows.sort(key=lambda r: -r[1])
print('top |AUC| vs sign(fwd_ret_1):')
for c, a in rows[:12]:
    print('   %-28s %.4f' % (c, a))
print('elapsed %.1fs  rows=%d' % (time.time()-t0, len(d)))
