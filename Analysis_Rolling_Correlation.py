import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ================= 配置 =================
ROLLING_WINDOW = 12  # 12周滚动窗口

# ================= 1. 数据加载与预处理 =================
def load_and_prep():
    try:
        df_pred_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        df_true_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        df_pred_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_true_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        
        for df in [df_pred_usa, df_true_usa, df_pred_chn, df_true_chn, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
        return df_pred_usa, df_true_usa, df_pred_chn, df_true_chn, sent_df
    except FileNotFoundError:
        print("❌ 缺少文件")
        exit()

def get_data_for_country(df_pred, df_true, raw_col, true_col, sent_df, country):
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

    # 先合并真实值的日期范围
    df_final = pd.merge(df_true[['date', true_col]], df_agg[['date', 'baseline']], on='date', how='left')
    df_final.rename(columns={true_col: 'truth'}, inplace=True)
    
    df_final = df_final.set_index('date')
    train_series = df_final['baseline'].dropna()
    full_idx = df_final.index
    future_idx = full_idx[full_idx > train_series.index.max()]
    
    if len(future_idx) > 0 and len(train_series) >= 52:
        pattern = train_series.iloc[-52:].values
        tiles = int(np.ceil(len(future_idx)/52))
        fill_values = np.tile(pattern, tiles)[:len(future_idx)]
        df_final.loc[future_idx, 'baseline'] = fill_values
    elif len(future_idx) > 0:
        df_final.loc[future_idx, 'baseline'] = train_series.mean()
        
    df_final['baseline'] = df_final['baseline'].interpolate()
    df_final = df_final.reset_index()

    sent_c = sent_df[sent_df['country'] == country][['date', 'sentiment_index']].copy()
    sent_c.columns = ['date', 'sentiment']
    
    df = pd.merge(df_final, sent_c, on='date', how='left')
    df['sentiment'] = df['sentiment'].interpolate()
    
    df = df.dropna(subset=['truth', 'baseline', 'sentiment']).copy()
    
    df['Panic'] = -1 * df['sentiment'] 
    df['Residual'] = df['truth'] - df['baseline']
    
    return df.sort_values('date')

# ================= 2. 滚动相关性分析 =================
def plot_rolling_correlation_full(df, country, ax):
    # 计算滚动相关
    rolling_corr = df['Panic'].rolling(window=ROLLING_WINDOW).corr(df['Residual'])
    
    # 绘图
    ax.plot(df['date'], rolling_corr, color='#1f77b4', linewidth=1.5, label='Dynamic Correlation')
    
    # 标记危机时刻 (红区)
    ax.fill_between(df['date'], 0, rolling_corr, 
                    where=(rolling_corr >= 0.3), 
                    color='red', alpha=0.3, label='Coupled Phase (Panic Drives Error)')
    
    # 标记脱钩时刻 (灰区)
    ax.fill_between(df['date'], 0, rolling_corr, 
                    where=(rolling_corr < 0.3), 
                    color='gray', alpha=0.1)
    
    # 标记 COVID NPIs 时期 (可选，辅助理解)
    # ax.axvspan(pd.to_datetime('2020-02-01'), pd.to_datetime('2022-12-01'), color='yellow', alpha=0.1, label='COVID NPIs')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(f'{country}: The Evolving Relationship (2015-2025)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_ylim(-1, 1)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 在图中标注
    ax.text(pd.to_datetime('2024-01-01'), 0.8, 'Post-COVID\nStrong Coupling', 
            color='darkred', fontweight='bold', ha='center', fontsize=10)

# ================= 主程序 =================
if __name__ == "__main__":
    df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df = load_and_prep()
    
    print("构建全量数据集...")
    data_usa = get_data_for_country(df_p_usa, df_t_usa, 'yhat', 'num_inc', sent_df, 'USA')
    data_chn = get_data_for_country(df_p_chn, df_t_chn, 'pCN_weighted', 'national_ili_weighted', sent_df, 'CHN')
    
    print(f"USA 数据范围: {data_usa['date'].min().date()} - {data_usa['date'].max().date()}")
    print(f"CHN 数据范围: {data_chn['date'].min().date()} - {data_chn['date'].max().date()}")
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    plot_rolling_correlation_full(data_usa, 'USA', ax1)
    plot_rolling_correlation_full(data_chn, 'China', ax2)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.set_xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2025-06-01'))
    
    plt.tight_layout()
    plt.savefig('./result/Analysis_Rolling_Correlation_Full.png', dpi=300)
    print("✅ 全周期动态相关性图已保存: Analysis_Rolling_Correlation_Full.png")