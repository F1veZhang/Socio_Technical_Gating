import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# ================= 配置 =================
BOOTSTRAP_ROUNDS = 30   # 稍微减少次数以加快速度
CCM_E = 3
CCM_TAU = 1
PHASE_3_START = '2022-12-07'

def load_and_prep_data():
    try:
        df_p_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        df_t_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        df_p_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_t_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        
        for df in [df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
        return df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df
    except:
        return None

def get_clean_series_diff(df_p, df_t, raw_col, true_col, sent_df, country):
    # 1. 常规处理
    if raw_col == 'pCN_weighted':
        df_p['raw'] = df_p['pCN'] if 'pCN' in df_p.columns else df_p['pS']*0.596 + df_p['pN']*0.404
    else:
        df_p['raw'] = df_p[raw_col]
        
    df_agg = df_p.groupby('date')['raw'].mean().reset_index()
    merged = pd.merge(df_agg, df_t[['date', true_col]], on='date').dropna()
    mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
    mu_t, sigma_t = merged[true_col].mean(), merged[true_col].std()
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    
    df_final = pd.merge(df_t[['date', true_col]], df_agg[['date', 'baseline']], on='date', how='left')
    df_final.rename(columns={true_col: 'truth'}, inplace=True)
    
    sent_c = sent_df[sent_df['country'] == country][['date', 'sentiment_index']].copy()
    sent_c.rename(columns={'sentiment_index': 'sentiment'}, inplace=True)
    
    df = pd.merge_asof(df_final.sort_values('date'), sent_c.sort_values('date'), 
                       on='date', direction='nearest', tolerance=pd.Timedelta(days=7))
    
    df['baseline'] = df['baseline'].interpolate()
    df['sentiment'] = df['sentiment'].interpolate()
    
    # Post-COVID 切片
    df_post = df[df['date'] >= PHASE_3_START].dropna().copy()
    
    # 2. 计算 Panic 和 Residual
    df_post['Panic'] = -1 * df_post['sentiment']
    df_post['Residual'] = df_post['truth'] - df_post['baseline']
    
    # === 关键步骤：一阶差分 (First Difference) ===
    df_post['Panic_Diff'] = df_post['Panic'].diff()
    df_post['Residual_Diff'] = df_post['Residual'].diff()
    
    # 差分后第一行是 NaN，需要去掉
    df_model = df_post.dropna(subset=['Panic_Diff', 'Residual_Diff']).copy()
    
    print(f"[{country}] Data Points after Differencing: {len(df_model)}")
    
    # Z-score 标准化
    X = (df_model['Panic_Diff'] - df_model['Panic_Diff'].mean()) / df_model['Panic_Diff'].std()
    Y = (df_model['Residual_Diff'] - df_model['Residual_Diff'].mean()) / df_model['Residual_Diff'].std()
    
    return X.values, Y.values

# CCM 核心
def ccm_core(data_from, data_to, L, E, tau):
    N = len(data_from)
    manifold, targets = [], []
    for t in range((E-1)*tau, N):
        manifold.append([data_from[t - k*tau] for k in range(E)])
        targets.append(data_to[t])
    
    manifold = np.array(manifold)
    targets = np.array(targets)
    if L > len(manifold): L = len(manifold)
    
    indices = np.arange(len(manifold))
    lib_idx = np.random.choice(indices, size=L, replace=True)
    lib_manifold = manifold[lib_idx]
    lib_targets = targets[lib_idx]
    
    dists = cdist(manifold, lib_manifold, metric='euclidean')
    preds = []
    
    for i in range(len(manifold)):
        d_i = dists[i]
        sorted_idx = np.argsort(d_i)
        valid_neighbors = []
        for idx in sorted_idx:
            if dists[i][idx] > 1e-5: valid_neighbors.append(idx)
            if len(valid_neighbors) == E+1: break
        if len(valid_neighbors) < E+1: valid_neighbors = sorted_idx[1:E+2]

        n_dists = d_i[valid_neighbors]
        n_targets = lib_targets[valid_neighbors]
        
        min_d = n_dists[0] if n_dists[0] > 1e-6 else 1e-6
        weights = np.exp(-n_dists / min_d)
        weights /= weights.sum()
        preds.append(np.dot(weights, n_targets))
        
    c = np.corrcoef(targets, preds)[0, 1]
    return c if not np.isnan(c) else 0

if __name__ == "__main__":
    raw_data = load_and_prep_data()
    if raw_data:
        df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df = raw_data
        
        # 使用差分数据
        X_us, Y_us = get_clean_series_diff(df_p_usa, df_t_usa, 'yhat', 'num_inc', sent_df, 'USA')
        X_cn, Y_cn = get_clean_series_diff(df_p_chn, df_t_chn, 'pCN_weighted', 'national_ili_weighted', sent_df, 'CHN')
        
        for country, X, Y in [('USA', X_us, Y_us), ('China', X_cn, Y_cn)]:
            max_L = len(X) - 5
            L_values = [15, max_L] 
            
            print(f"\n>>> {country} Differenced CCM (Strict Test)")
            for L in L_values:
                # Dir 1: Panic -> Epi
                rhos1 = [ccm_core(Y, X, L, CCM_E, CCM_TAU) for _ in range(BOOTSTRAP_ROUNDS)]
                mean1 = np.mean(rhos1)
                
                # Dir 2: Epi -> Panic
                rhos2 = [ccm_core(X, Y, L, CCM_E, CCM_TAU) for _ in range(BOOTSTRAP_ROUNDS)]
                mean2 = np.mean(rhos2)
                
                print(f"  [L={L}] Panic->Epi: {mean1:.3f} | Epi->Panic: {mean2:.3f}")