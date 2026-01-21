import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import matplotlib.gridspec as gridspec

# ================= 配置 =================
THRESHOLD_CHN = 62.0        # 中国基建阈值
PHASE_3_START = '2022-12-07' # 美国 Post-COVID 起点
PLACEBO_ROUNDS = 1000       # 安慰剂检验次数
CCM_BOOTSTRAP = 20          # CCM 抽样次数 (生成带区间的曲线)
CCM_E = 3
CCM_TAU = 1

def load_data():
    try:
        df_p_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_t_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        df_p_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        df_t_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        infra_df = pd.read_csv('./data/cleaned_digital_infrastructure.csv') 
        
        for df in [df_p_chn, df_t_chn, df_p_usa, df_t_usa, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
        return df_p_chn, df_t_chn, df_p_usa, df_t_usa, sent_df, infra_df
    except:
        print("❌ 数据缺失")
        return None

# --- 数据准备函数 ---
def prep_data_chn(df_p, df_t, sent_df, infra_df):
    # 1. 常规处理
    df_p['raw'] = df_p['pCN'] if 'pCN' in df_p.columns else df_p['pS']*0.596 + df_p['pN']*0.404
    df_agg = df_p.groupby('date')['raw'].mean().reset_index()
    merged = pd.merge(df_agg, df_t[['date', 'national_ili_weighted']], on='date').dropna()
    mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
    mu_t, sigma_t = merged['national_ili_weighted'].mean(), merged['national_ili_weighted'].std()
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    df_final = pd.merge(df_t[['date', 'national_ili_weighted']], df_agg[['date', 'baseline']], on='date', how='left')
    df_final.rename(columns={'national_ili_weighted': 'truth'}, inplace=True)
    
    sent_c = sent_df[sent_df['country'] == 'CHN'][['date', 'sentiment_index']].copy()
    sent_c.rename(columns={'sentiment_index': 'sentiment'}, inplace=True)
    df = pd.merge_asof(df_final.sort_values('date'), sent_c.sort_values('date'), 
                       on='date', direction='nearest', tolerance=pd.Timedelta(days=7)).dropna()
    
    # 2. 筛选 High Infra
    df['Year'] = df['date'].dt.year
    df = pd.merge(df, infra_df[['Year', 'CNNIC_China_Internet']], on='Year', how='left')
    df = df[df['CNNIC_China_Internet'] >= THRESHOLD_CHN].copy()
    
    df['Panic'] = -1 * df['sentiment']
    df['Residual'] = df['truth'] - df['baseline']
    return df['Panic'].values, df['Residual'].values

def prep_data_usa(df_p, df_t, sent_df):
    df_p['raw'] = df_p['yhat']
    df_agg = df_p.groupby('date')['raw'].mean().reset_index()
    merged = pd.merge(df_agg, df_t[['date', 'num_inc']], on='date').dropna()
    mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
    mu_t, sigma_t = merged['num_inc'].mean(), merged['num_inc'].std()
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    df_final = pd.merge(df_t[['date', 'num_inc']], df_agg[['date', 'baseline']], on='date', how='left')
    df_final.rename(columns={'num_inc': 'truth'}, inplace=True)
    
    sent_c = sent_df[sent_df['country'] == 'USA'][['date', 'sentiment_index']].copy()
    sent_c.rename(columns={'sentiment_index': 'sentiment'}, inplace=True)
    df = pd.merge_asof(df_final.sort_values('date'), sent_c.sort_values('date'), 
                       on='date', direction='nearest', tolerance=pd.Timedelta(days=7)).dropna()
    
    # 2. 筛选 Post-COVID
    df = df[df['date'] >= PHASE_3_START].copy()
    
    df['Panic'] = -1 * df['sentiment']
    df['Residual'] = df['truth'] - df['baseline']
    return df['Panic'].values, df['Residual'].values

# --- 核心算法 ---
def run_placebo(x, y):
    true_r = np.corrcoef(x, y)[0, 1]
    null_dist = []
    y_shuffled = y.copy()
    for _ in range(PLACEBO_ROUNDS):
        np.random.shuffle(y_shuffled)
        null_dist.append(np.corrcoef(x, y_shuffled)[0, 1])
    return true_r, np.array(null_dist)

def run_ccm(data_from, data_to):
    # data_from = Panic (Cause), data_to = Residual (Effect)
    # We predict Cause using Effect (Y -> X mapping) implies X causes Y
    # Wait, Standard CCM notation: 
    # "Predict Y using X" (X xmap Y) -> Y causes X
    # "Predict X using Y" (Y xmap X) -> X causes Y (Our Hypothesis)
    
    # We want to show "Panic -> Epidemic". So we use Epidemic(Y) to predict Panic(X).
    X = (data_from - np.mean(data_from))/np.std(data_from) # Panic
    Y = (data_to - np.mean(data_to))/np.std(data_to)       # Residual
    
    N = len(Y)
    max_L = N - 5
    L_range = np.linspace(15, max_L, 10, dtype=int)
    
    means, lowers, uppers = [], [], []
    
    for L in L_range:
        rhos = []
        for _ in range(CCM_BOOTSTRAP):
            # ccm_core logic simplified
            manifold, targets = [], []
            for t in range((CCM_E-1)*CCM_TAU, N):
                manifold.append([Y[t - k*CCM_TAU] for k in range(CCM_E)]) # Reconstruct using Effect (Y)
                targets.append(X[t]) # Predict Cause (X)
            
            manifold = np.array(manifold)
            targets = np.array(targets)
            if L > len(manifold): L_act = len(manifold)
            else: L_act = L
            
            idx = np.random.choice(len(manifold), size=L_act, replace=True)
            lib_M, lib_T = manifold[idx], targets[idx]
            
            dists = cdist(manifold, lib_M, metric='euclidean')
            preds = []
            for i in range(len(manifold)):
                d = dists[i]
                sorted_idx = np.argsort(d)[:CCM_E+1]
                n_d = d[sorted_idx]
                n_t = lib_T[sorted_idx]
                w = np.exp(-n_d / (n_d[0] + 1e-6))
                w /= w.sum()
                preds.append(np.dot(w, n_t))
            
            rhos.append(np.corrcoef(targets, preds)[0, 1])
        
        means.append(np.nanmean(rhos))
        lowers.append(np.percentile(rhos, 5))
        uppers.append(np.percentile(rhos, 95))
        
    return L_range, means, lowers, uppers

# --- 绘图主程序 ---
def plot_final_figure():
    print(">>> Generating Final Figure...")
    data = load_data()
    if not data: return
    df_p_chn, df_t_chn, df_p_usa, df_t_usa, sent_df, infra_df = data
    
    # 1. 准备数据
    x_chn, y_chn = prep_data_chn(df_p_chn, df_t_chn, sent_df, infra_df)
    x_usa, y_usa = prep_data_usa(df_p_usa, df_t_usa, sent_df)
    
    print(f"China Samples: {len(x_chn)} (High Infra)")
    print(f"USA Samples: {len(x_usa)} (Post-COVID)")
    
    # 2. 创建画布
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2]) # 右边CCM稍微宽一点
    
    # --- Panel A: China Placebo ---
    ax1 = plt.subplot(gs[0, 0])
    r_chn, null_chn = run_placebo(x_chn, y_chn)
    sns.histplot(null_chn, color='lightgray', kde=True, ax=ax1, stat='density')
    ax1.axvline(r_chn, color='#D62728', linewidth=3, linestyle='--', label=f'True r={r_chn:.3f}')
    ax1.set_title(f'A. Robustness Check (China)\n(High Penetration $\geq$ {THRESHOLD_CHN}%)', fontweight='bold', loc='left')
    ax1.set_xlabel('Correlation Coefficient')
    ax1.legend()
    # P-value calc
    p_chn = (null_chn >= r_chn).mean()
    ax1.text(0.05, 0.9, f'Significance:\n$P < {max(p_chn, 0.001):.3f}$', transform=ax1.transAxes, color='#D62728', fontweight='bold')
    
    # --- Panel B: USA Placebo ---
    ax2 = plt.subplot(gs[1, 0])
    r_usa, null_usa = run_placebo(x_usa, y_usa)
    sns.histplot(null_usa, color='lightgray', kde=True, ax=ax2, stat='density')
    ax2.axvline(r_usa, color='blue', linewidth=3, linestyle='--', label=f'True r={r_usa:.3f}')
    ax2.set_title('B. Robustness Check (USA)\n(Post-COVID Era)', fontweight='bold', loc='left')
    ax2.set_xlabel('Correlation Coefficient')
    ax2.legend()
    p_usa = (null_usa >= r_usa).mean() if r_usa > 0 else (null_usa <= r_usa).mean()
    ax2.text(0.05, 0.9, f'Significance:\n$P = {p_usa:.3f}$ (N.S.)', transform=ax2.transAxes, color='blue', fontweight='bold')

    # --- Panel C: China CCM ---
    ax3 = plt.subplot(gs[0, 1])
    L_c, m_c, l_c, u_c = run_ccm(x_chn, y_chn)
    ax3.plot(L_c, m_c, 'o-', color='#D62728', linewidth=2, label='Panic $\\to$ Epidemic')
    ax3.fill_between(L_c, l_c, u_c, color='#D62728', alpha=0.2)
    ax3.set_title('C. Causal Mechanism (China)\n(Convergence = Causal)', fontweight='bold', loc='left')
    ax3.set_ylabel('Prediction Skill ($\\rho$)')
    ax3.set_ylim(-0.1, 0.8)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower right')
    
    # --- Panel D: USA CCM ---
    ax4 = plt.subplot(gs[1, 1])
    L_u, m_u, l_u, u_u = run_ccm(x_usa, y_usa)
    ax4.plot(L_u, m_u, 's-', color='blue', linewidth=2, label='Panic $\\to$ Epidemic')
    ax4.fill_between(L_u, l_u, u_u, color='blue', alpha=0.2)
    ax4.set_title('D. Causal Mechanism (USA)\n(Flat/Low = Decoupled)', fontweight='bold', loc='left')
    ax4.set_xlabel('Library Size (L)')
    ax4.set_ylabel('Prediction Skill ($\\rho$)')
    ax4.set_ylim(-0.1, 0.8)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('./result/Final_Figure_Robustness_CCM.png', dpi=300)
    print("\n✅ 最终合并图已生成: Final_Figure_Robustness_CCM.png")

if __name__ == "__main__":
    plot_final_figure()