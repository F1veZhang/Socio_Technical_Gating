import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================= 1. 数据加载 =================
try:
    df_pred_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
    df_pred_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
    df_true_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
    df_true_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
    sentiment_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')

    # 转换时间格式
    for df in [df_pred_chn, df_pred_usa, df_true_chn, df_true_usa, sentiment_df]:
        df['date'] = pd.to_datetime(df['date'])

except FileNotFoundError as e:
    print(f"❌ 错误: 找不到文件 {e.filename}")
    exit()

# ================= 2. 数据处理：提取“基于关注度”的基准线 =================

def prepare_baseline(df_pred, df_true, raw_col, true_col):
    # 1. 提取前作预测值
    if raw_col == 'pCN_weighted':
        if 'pCN' in df_pred.columns:
            df_pred['raw'] = df_pred['pCN']
        else:
            df_pred['raw'] = df_pred['pS'] * 0.596 + df_pred['pN'] * 0.404
    else:
        df_pred['raw'] = df_pred[raw_col]
        
    # 2. 聚合去重
    df_agg = df_pred.groupby('date')['raw'].mean().reset_index()
    
    # 3. 均值方差匹配 (Rescaling)
    merged = pd.merge(df_agg, df_true[['date', true_col]], on='date').dropna()
    if len(merged) > 10:
        mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
        mu_t, sigma_t = merged[true_col].mean(), merged[true_col].std()
    else:
        mu_p, sigma_p = df_agg['raw'].mean(), df_agg['raw'].std()
        mu_t, sigma_t = df_true[true_col].mean(), df_true[true_col].std()
        
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    df_agg['baseline'] = df_agg['baseline'].apply(lambda x: max(0, x))
    
    return df_agg.set_index('date')['baseline']

series_base_chn = prepare_baseline(df_pred_chn, df_true_chn, 'pCN_weighted', 'national_ili_weighted')
series_base_usa = prepare_baseline(df_pred_usa, df_true_usa, 'yhat', 'num_inc')

# ================= 3. 无缝外推 (Extrapolation) =================

def extend_baseline_seamless(train_series, target_dates):
    target_dates = np.unique(target_dates)
    full_series = pd.Series(index=target_dates, dtype=float)
    
    common = train_series.index.intersection(full_series.index)
    full_series.loc[common] = train_series.loc[common]
    
    last_date = train_series.index.max()
    future_idx = full_series.index[full_series.index > last_date]
    
    if len(future_idx) > 0:
        if len(train_series) >= 52:
            pattern = train_series.iloc[-52:].values
            tiles = int(np.ceil(len(future_idx) / 52))
            fill = np.tile(pattern, tiles)[:len(future_idx)]
            full_series.loc[future_idx] = fill
        else:
            full_series.loc[future_idx] = train_series.mean()
            
    return full_series.interpolate(method='time')

dates_chn = df_true_chn['date'].sort_values().unique()
dates_usa = df_true_usa['date'].sort_values().unique()

full_base_chn = extend_baseline_seamless(series_base_chn, dates_chn)
full_base_usa = extend_baseline_seamless(series_base_usa, dates_usa)

# ================= 4. 构建最终数据集 =================

def build_dataset(df_true, pred_series, true_col, country_code):
    df = df_true.copy().sort_values('date')
    df['baseline'] = pred_series.values
    df[true_col] = df[true_col].interpolate()
    
    sent_country = sentiment_df[sentiment_df['country'] == country_code][['date', 'sentiment_index']]
    df = pd.merge(df, sent_country, on='date', how='left')
    if 'sentiment_index_y' in df.columns:
        df['sentiment_index'] = df['sentiment_index_y'].combine_first(df['sentiment_index_x'])
    
    df['sentiment_smooth'] = df['sentiment_index'].interpolate().rolling(window=8, center=True).mean()
    df['deviation'] = df[true_col] - df['baseline']
    return df

df_final_chn = build_dataset(df_true_chn, full_base_chn, 'national_ili_weighted', 'CHN')
df_final_usa = build_dataset(df_true_usa, full_base_usa, 'num_inc', 'USA')

# ================= 5. 绘图 (Volume vs Valence + Full Areas) =================

def plot_volume_valence_full(data, country_name, true_col, title, filename):
    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # --- Top Plot: Volume-Based Baseline vs Reality ---
    ax1.plot(data['date'], data[true_col], label='Observed Reality (Surveillance Data)', color='black', linewidth=2)
    ax1.plot(data['date'], data['baseline'], label='Volume-Based Baseline (Previous Work)', color='#1F77B4', linestyle='--', linewidth=2)
    
    # 标记预测起始点 (2024)
    split_date = pd.to_datetime('2023-12-25')
    ax1.axvline(split_date, color='gray', linestyle=':', linewidth=2)
    ax1.text(split_date, ax1.get_ylim()[1]*0.95, '  Prediction Phase (2024)', color='gray', fontweight='bold')
    
    # 上图也填充一点颜色，辅助视觉
    ax1.fill_between(data['date'], data[true_col], data['baseline'], 
                     where=(data[true_col] > data['baseline']), 
                     interpolate=True, color='red', alpha=0.1, label='Excess Outbreak')
    
    ax1.set_ylabel('Epidemic Intensity', fontsize=12, fontweight='bold')
    ax1.set_title(f'A. {title}: Limitations of Volume-Based Surveillance', loc='left', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Bottom Plot: Deviation vs Sentiment ---
    ax2_twin = ax2.twinx()
    
    # 红色区域：Volume模型无法解释的偏差 (True > Baseline)
    ax2.fill_between(data['date'], 0, data['deviation'], where=(data['deviation']>0), 
                     interpolate=True, color='red', alpha=0.4, label='Under-prediction (Excess Cases)')

    # 绿色区域：模型高估的部分 (True < Baseline) -> 这就是你要保留的！
    ax2.fill_between(data['date'], 0, data['deviation'], where=(data['deviation']<=0), 
                     interpolate=True, color='green', alpha=0.2, label='Over-prediction (False Alarm)')
    
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Deviation Magnitude', fontsize=12, fontweight='bold')
    
    # 情感曲线
    ax2_twin.plot(data['date'], data['sentiment_smooth'], color='#D62728', linewidth=3, label='Sentiment Valence (Smoothed)')
    ax2_twin.set_ylabel('Social Sentiment Valence', color='#D62728', fontsize=12, fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor='#D62728')
    
    ax2.set_title(f'B. {title}: Sentiment Valence Explains the Gap', loc='left', fontsize=16, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # 图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='red', alpha=0.4, label='Under-prediction (Excess Cases)'),
        Patch(facecolor='green', alpha=0.2, label='Over-prediction (False Alarm)'),
        Line2D([0], [0], color='#D62728', lw=3, label='Sentiment Valence (Smoothed)')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', frameon=True)
    
    ax2.set_xlabel('Year', fontsize=14)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.set_xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2025-06-01'))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ 图表已保存: {filename}")

# 生成图表
plot_volume_valence_full(df_final_chn, 'China', 'national_ili_weighted', 'China', './result/Mechanism_China_VolumeValence_Full.png')
plot_volume_valence_full(df_final_usa, 'USA', 'num_inc', 'USA', './result/Mechanism_USA_VolumeValence_Full.png')