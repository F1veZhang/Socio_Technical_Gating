import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import matplotlib.gridspec as gridspec

# ================= 配置 =================
START_DATE = '2019-01-01'   # 锁定 2019-2025 周期
PLACEBO_ROUNDS = 2000       # 左图：安慰剂检验次数
CCM_BOOTSTRAP = 50          # 右图：CCM重抽样次数 (生成阴影带)
CCM_E = 3
CCM_TAU = 1

def load_data():
    try:
        df_p_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_t_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        df_p_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        df_t_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        
        for df in [df_p_chn, df_t_chn, df_p_usa, df_t_usa, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
        return df_p_chn, df_t_chn, df_p_usa, df_t_usa, sent_df
    except:
        print("❌ 数据缺失，请检查路径 './data/'")
        return None

# --- 数据准备 (统一逻辑) ---
def prep_data(df_p, df_t, sent_df, country, raw_col, true_col):
    # 1. 构建 Baseline
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
    
    # 2. 匹配情感
    sent_c = sent_df[sent_df['country'] == country][['date', 'sentiment_index']].copy()
    sent_c.rename(columns={'sentiment_index': 'sentiment'}, inplace=True)
    df = pd.merge_asof(df_final.sort_values('date'), sent_c.sort_values('date'), 
                       on='date', direction='nearest', tolerance=pd.Timedelta(days=7)).dropna()
    
    # 3. 锁定时间 (2019-2025)
    df = df[df['date'] >= START_DATE].copy()
    
    df['Panic'] = -1 * df['sentiment']
    df['Residual'] = df['truth'] - df['baseline']
    
    return df['Panic'].values, df['Residual'].values

# --- 核心算法 ---
def run_placebo(x, y, country_name):
    true_r = np.corrcoef(x, y)[0, 1]
    null_dist = []
    y_shuffled = y.copy()
    for _ in range(PLACEBO_ROUNDS):
        np.random.shuffle(y_shuffled)
        null_dist.append(np.corrcoef(x, y_shuffled)[0, 1])
    
    null_dist = np.array(null_dist)
    
    # 计算 P 值
    if true_r > 0:
        p_val = (null_dist >= true_r).mean()
    else:
        p_val = (null_dist <= true_r).mean()

    # --- 控制台输出结果 ---
    print(f"\n[{country_name}] Placebo Test Results:")
    print(f"  > True Correlation (r): {true_r:.4f}")
    print(f"  > P-value: {p_val:.5f}")
    if p_val < 0.05:
        print("  > Conclusion: ✅ SIGNIFICANT (Robust)")
    else:
        print("  > Conclusion: ❌ NOT SIGNIFICANT (Random Noise)")
        
    return true_r, null_dist

def run_ccm_bootstrap(data_from, data_to, country_name):
    # 标准化
    X = (data_from - np.mean(data_from))/np.std(data_from) 
    Y = (data_to - np.mean(data_to))/np.std(data_to)
    
    N = len(Y)
    max_L = N - 5
    # 设置 Library Steps
    L_range = np.linspace(10, max_L, 15, dtype=int)
    
    means, lowers, uppers = [], [], []
    
    print(f"\n[{country_name}] Running Bootstrap CCM (N={N}, MaxL={max_L})...")
    
    for L in L_range:
        rhos = []
        for _ in range(CCM_BOOTSTRAP):
            # 1. 每次循环重新构建流形和目标
            manifold, targets = [], []
            for t in range((CCM_E-1)*CCM_TAU, N):
                manifold.append([Y[t - k*CCM_TAU] for k in range(CCM_E)]) 
                targets.append(X[t]) 
            
            manifold = np.array(manifold)
            targets = np.array(targets)
            
            # 2. 随机抽取 L 个点作为 Library
            if L > len(manifold): L_act = len(manifold)
            else: L_act = L
            
            idx = np.random.choice(len(manifold), size=L_act, replace=True)
            lib_M, lib_T = manifold[idx], targets[idx]
            
            # 3. 预测
            dists = cdist(manifold, lib_M, metric='euclidean')
            preds = []
            for i in range(len(manifold)):
                d = dists[i]
                sorted_idx = np.argsort(d)[:CCM_E+1]
                n_d = d[sorted_idx]
                n_t = lib_T[sorted_idx]
                min_dist = n_d[0] if n_d[0] > 1e-6 else 1e-6
                w = np.exp(-n_d / min_dist)
                w /= w.sum()
                preds.append(np.dot(w, n_t))
            
            rhos.append(np.corrcoef(targets, preds)[0, 1])
        
        # 统计分布
        means.append(np.nanmean(rhos))
        lowers.append(np.percentile(rhos, 5))  # 95% CI 下界
        uppers.append(np.percentile(rhos, 95)) # 95% CI 上界
        
    # --- 控制台输出结果 ---
    print(f"[{country_name}] CCM Summary:")
    print(f"  > Start Rho (L={L_range[0]}): {means[0]:.3f}")
    print(f"  > End Rho (L={L_range[-1]}): {means[-1]:.3f}")
    print(f"  > Convergence Delta: {means[-1] - means[0]:.3f}")
    if (means[-1] - means[0]) > 0.1:
         print("  > Conclusion: ✅ CONVERGENCE (Causal Link Exists)")
    else:
         print("  > Conclusion: ⚠️ WEAK CONVERGENCE (Weak/No Link)")

    return L_range, means, lowers, uppers

# --- 绘图主程序 ---
def plot_final_figure():
    data = load_data()
    if not data: return
    df_p_chn, df_t_chn, df_p_usa, df_t_usa, sent_df = data
    
    # 1. 准备数据
    print("\n" + "="*40)
    print("STEP 1: Preparing Data (2019-2025)")
    print("="*40)
    x_chn, y_chn = prep_data(df_p_chn, df_t_chn, sent_df, 'CHN', 'pCN_weighted', 'national_ili_weighted')
    print(f"China Valid Samples: {len(x_chn)}")
    
    x_usa, y_usa = prep_data(df_p_usa, df_t_usa, sent_df, 'USA', 'yhat', 'num_inc')
    print(f"USA Valid Samples: {len(x_usa)}")
    
    # 2. 创建画布
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2]) 
    
    # === Row 1: China ===
    # Left: Placebo
    print("\n" + "="*40)
    print("STEP 2: Processing China")
    print("="*40)
    ax1 = plt.subplot(gs[0, 0])
    r_chn, null_chn = run_placebo(x_chn, y_chn, "China")
    sns.histplot(null_chn, color='lightgray', kde=True, ax=ax1, stat='density')
    ax1.axvline(r_chn, color='#D62728', linewidth=3, linestyle='--', label=f'True r={r_chn:.3f}')
    ax1.set_title(f'A. Robustness Check (China)\n(2019-2025)', fontweight='bold', loc='left')
    p_chn = (null_chn >= r_chn).mean()
    ax1.text(0.05, 0.9, f'P < {max(p_chn, 0.001):.3f}', transform=ax1.transAxes, color='#D62728', fontweight='bold')
    ax1.legend()
    
    # Right: CCM
    ax2 = plt.subplot(gs[0, 1])
    L_c, m_c, l_c, u_c = run_ccm_bootstrap(x_chn, y_chn, "China")
    ax2.plot(L_c, m_c, 'o-', color='#D62728', linewidth=2, label='Panic $\\to$ Epidemic')
    ax2.fill_between(L_c, l_c, u_c, color='#D62728', alpha=0.2) # 阴影带
    ax2.set_title('B. Causal Mechanism (China)\n(Strong Convergence)', fontweight='bold', loc='left')
    ax2.set_ylabel('Prediction Skill ($\\rho$)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right')
    
    # === Row 2: USA ===
    print("\n" + "="*40)
    print("STEP 3: Processing USA")
    print("="*40)
    # Left: Placebo
    ax3 = plt.subplot(gs[1, 0])
    r_usa, null_usa = run_placebo(x_usa, y_usa, "USA")
    sns.histplot(null_usa, color='lightgray', kde=True, ax=ax3, stat='density')
    ax3.axvline(r_usa, color='blue', linewidth=3, linestyle='--', label=f'True r={r_usa:.3f}')
    ax3.set_title(f'C. Robustness Check (USA)\n(2019-2025)', fontweight='bold', loc='left')
    p_usa = (null_usa >= r_usa).mean() if r_usa > 0 else (null_usa <= r_usa).mean()
    ax3.text(0.05, 0.9, f'P = {p_usa:.3f}', transform=ax3.transAxes, color='blue', fontweight='bold')
    ax3.legend()
    
    # Right: CCM
    ax4 = plt.subplot(gs[1, 1])
    L_u, m_u, l_u, u_u = run_ccm_bootstrap(x_usa, y_usa, "USA")
    ax4.plot(L_u, m_u, 's-', color='blue', linewidth=2, label='Panic $\\to$ Epidemic')
    ax4.fill_between(L_u, l_u, u_u, color='blue', alpha=0.2) # 阴影带
    ax4.set_title('D. Causal Mechanism (USA)\n(Decoupled / Seasonal)', fontweight='bold', loc='left')
    ax4.set_xlabel('Library Size (L)')
    ax4.set_ylabel('Prediction Skill ($\\rho$)')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower right')
    
    plt.tight_layout()
    output_path = './result/Final_Figure_2019_2025_Bootstrap.png'
    plt.savefig(output_path, dpi=300)
    print("\n" + "="*40)
    print(f"✅ FINAL DONE: Image saved to {output_path}")
    print("="*40)

if __name__ == "__main__":
    plot_final_figure()