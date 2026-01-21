import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ================= 配置 =================
START_DATE_POST_COVID = '2023-01-01'

# ================= 1. 数据加载与基准构建 =================
def load_and_prep():
    print("正在加载数据...")
    try:
        df_pred_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        df_true_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        df_pred_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_true_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        
        # 修复点：加载 weekly_sentiment_series_FINAL.csv，确保有 sentiment_index
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv') 
        
        # 时间格式转换
        for df in [df_pred_usa, df_true_usa, df_pred_chn, df_true_chn, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
            
        print("✅ 数据加载完成")
        return df_pred_usa, df_true_usa, df_pred_chn, df_true_chn, sent_df
    except FileNotFoundError as e:
        print(f"❌ 缺少文件: {e.filename}")
        exit()

def build_baseline(df_pred, df_true, raw_col, true_col):
    # 提取预测
    if raw_col == 'pCN_weighted':
        if 'pCN' in df_pred.columns:
            df_pred['raw'] = df_pred['pCN']
        else:
            df_pred['raw'] = df_pred['pS']*0.596 + df_pred['pN']*0.404
    else:
        df_pred['raw'] = df_pred[raw_col]
        
    df_agg = df_pred.groupby('date')['raw'].mean().reset_index()
    
    # Rescaling
    merged = pd.merge(df_agg, df_true[['date', true_col]], on='date').dropna()
    if len(merged) > 10:
        mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
        mu_t, sigma_t = merged[true_col].mean(), merged[true_col].std()
    else:
        mu_p, sigma_p = df_agg['raw'].mean(), df_agg['raw'].std()
        mu_t, sigma_t = df_true[true_col].mean(), df_true[true_col].std()
    
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    df_agg['baseline'] = df_agg['baseline'].apply(lambda x: max(0, x))
    
    # 拼接 & 插值
    df_final = pd.merge(df_true, df_agg[['date', 'baseline']], on='date', how='left')
    df_final.rename(columns={true_col: 'truth'}, inplace=True)
    
    # Extrapolation 2024-2025
    df_final = df_final.set_index('date')
    train = df_final['baseline'].dropna()
    full_idx = df_final.index
    future_idx = full_idx[full_idx > train.index.max()]
    
    if len(future_idx) > 0 and len(train) >= 52:
        pattern = train.iloc[-52:].values
        tiles = int(np.ceil(len(future_idx)/52))
        fill = np.tile(pattern, tiles)[:len(future_idx)]
        df_final.loc[future_idx, 'baseline'] = fill
    elif len(future_idx) > 0:
        df_final.loc[future_idx, 'baseline'] = train.mean()
        
    df_final['baseline'] = df_final['baseline'].interpolate()
    return df_final.reset_index()

# ================= 2. 训练两种修正模型 =================
def train_models(df, sent_df, country_code):
    print(f"\n>>> 分析国家: {country_code} <<<")
    
    # 1. 准备情感数据 (统一使用 sentiment_index)
    sent_c = sent_df[sent_df['country'] == country_code][['date', 'sentiment_index']].copy()
    sent_c.rename(columns={'sentiment_index': 'sentiment'}, inplace=True)
    
    df = pd.merge(df, sent_c, on='date', how='left')
    df['sentiment'] = df['sentiment'].interpolate()
    
    # 2. 筛选 Post-COVID
    mask = df['date'] >= START_DATE_POST_COVID
    df_post = df.loc[mask].dropna(subset=['truth', 'baseline', 'sentiment']).copy()
    
    if len(df_post) < 10:
        print("⚠️ 数据不足，跳过")
        return None
        
    df_post['residual'] = df_post['truth'] - df_post['baseline']
    
    # --- 方法 1: DLM (Linear Robust) ---
    # 特征: Sentiment + Lags
    df_post['sent_lag1'] = df_post['sentiment'].shift(1).fillna(0)
    df_post['sent_lag2'] = df_post['sentiment'].shift(2).fillna(0)
    
    X_dlm = df_post[['sentiment', 'sent_lag1', 'sent_lag2']]
    y = df_post['residual']
    
    model_dlm = Ridge(alpha=1.0)
    model_dlm.fit(X_dlm, y)
    df_post['pred_dlm'] = df_post['baseline'] + model_dlm.predict(X_dlm)
    
    # --- 方法 2: NLI (Non-Linear Interaction) ---
    # 特征: Sentiment, Sentiment^2, Interaction
    df_post['sent_sq'] = df_post['sentiment'] ** 2
    df_post['interaction'] = df_post['sentiment'] * df_post['baseline']
    
    X_nli = df_post[['sentiment', 'sent_sq', 'interaction', 'sent_lag1']]
    
    model_nli = Ridge(alpha=0.5) 
    model_nli.fit(X_nli, y)
    df_post['pred_nli'] = df_post['baseline'] + model_nli.predict(X_nli)
    
    # 计算 RMSE
    rmse_base = np.sqrt(mean_squared_error(df_post['truth'], df_post['baseline']))
    rmse_dlm = np.sqrt(mean_squared_error(df_post['truth'], df_post['pred_dlm']))
    rmse_nli = np.sqrt(mean_squared_error(df_post['truth'], df_post['pred_nli']))
    
    print(f"Baseline RMSE: {rmse_base:,.0f}")
    print(f"DLM (Linear) RMSE: {rmse_dlm:,.0f} (Imp: {(rmse_base-rmse_dlm)/rmse_base*100:.1f}%)")
    print(f"NLI (Non-Linear) RMSE: {rmse_nli:,.0f} (Imp: {(rmse_base-rmse_nli)/rmse_base*100:.1f}%)")
    
    return df_post

# ================= 3. 绘图 (4 Lines) =================
def plot_4_lines(df, country, filename):
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 1. Truth (Black)
    ax.plot(df['date'], df['truth'], color='black', linewidth=3, label='Observed Reality (Gold Standard)', alpha=0.8, zorder=10)
    
    # 2. Baseline (Gray Dashed)
    ax.plot(df['date'], df['baseline'], color='gray', linestyle='--', linewidth=2, label='Baseline (Volume-Based)', alpha=0.6)
    
    # 3. DLM (Blue)
    ax.plot(df['date'], df['pred_dlm'], color='#1f77b4', linestyle='-', linewidth=2, label='Method 1: Linear Correction (DLM)', alpha=0.8)
    
    # 4. NLI (Red) - The "Better" One
    ax.plot(df['date'], df['pred_nli'], color='#d62728', linestyle='-', linewidth=3, label='Method 2: Non-Linear Interaction (Ours)', alpha=0.9)
    
    ax.set_title(f'Hierarchical Correction Analysis: {country} (2023-2025)\nFrom Linear to Non-Linear Dynamics', loc='left', fontsize=16, fontweight='bold')
    ax.set_ylabel('Epidemic Intensity', fontsize=12)
    ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.95)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved: {filename}")

# ================= 主程序 =================
if __name__ == "__main__":
    df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df = load_and_prep()
    
    # 构建基准
    base_usa = build_baseline(df_p_usa, df_t_usa, 'yhat', 'num_inc')
    base_chn = build_baseline(df_p_chn, df_t_chn, 'pCN_weighted', 'national_ili_weighted')
    
    # 训练与绘图
    res_usa = train_models(base_usa, sent_df, 'USA')
    if res_usa is not None:
        plot_4_lines(res_usa, 'USA', './result/Final_4Lines_USA_23-25.png')
    
    res_chn = train_models(base_chn, sent_df, 'CHN')
    if res_chn is not None:
        plot_4_lines(res_chn, 'China', './result/Final_4Lines_China_23-25.png')