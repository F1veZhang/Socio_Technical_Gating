import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ================= 1. 数据加载 =================
try:
    df_pred_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
    df_pred_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
    df_true_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
    df_true_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
    sentiment_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
    
    for df in [df_pred_chn, df_pred_usa, df_true_chn, df_true_usa, sentiment_df]:
        df['date'] = pd.to_datetime(df['date'])

except FileNotFoundError:
    print("❌ 请确保所有数据文件都在当前目录下")
    exit()

# --- 辅助函数：构建分析数据集 ---
def build_analysis_df(df_pred, df_true, raw_col, true_col, country_code):
    # 1. 提取原始预测并聚合
    if raw_col == 'pCN_weighted':
        if 'pCN' in df_pred.columns:
            df_pred['raw'] = df_pred['pCN']
        else:
            df_pred['raw'] = df_pred['pS']*0.596 + df_pred['pN']*0.404
    else:
        df_pred['raw'] = df_pred[raw_col]
        
    df_agg = df_pred.groupby('date')['raw'].mean().reset_index()
    
    # 2. Rescaling (均值方差匹配)
    merged = pd.merge(df_agg, df_true[['date', true_col]], on='date').dropna()
    if len(merged) > 10:
        mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
        mu_t, sigma_t = merged[true_col].mean(), merged[true_col].std()
    else:
        mu_p, sigma_p = df_agg['raw'].mean(), df_agg['raw'].std()
        mu_t, sigma_t = df_true[true_col].mean(), df_true[true_col].std()
    
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    df_agg['baseline'] = df_agg['baseline'].apply(lambda x: max(0, x))
    
    # 3. 拼接真实值
    df_final = pd.merge(df_true, df_agg[['date', 'baseline']], on='date', how='left')
    
    # 4. 无缝外推 Baseline
    df_final = df_final.set_index('date')  # <--- CRITICAL FIX
    
    train_series = df_final['baseline'].dropna()
    full_idx = df_final.index
    future_idx = full_idx[full_idx > train_series.index.max()]
    
    # 克隆最后一年模式
    if len(future_idx) > 0 and len(train_series) >= 52:
        pattern = train_series.iloc[-52:].values
        tiles = int(np.ceil(len(future_idx)/52))
        fill_values = np.tile(pattern, tiles)[:len(future_idx)]
        # 赋值给对应索引
        df_final.loc[future_idx, 'baseline'] = fill_values
    elif len(future_idx) > 0:
        df_final.loc[future_idx, 'baseline'] = train_series.mean()
    
    # 插值填补微小空洞
    df_final['baseline'] = df_final['baseline'].interpolate()
    df_final = df_final.reset_index() # 恢复索引
    
    # 5. 匹配情感
    sent_c = sentiment_df[sentiment_df['country'] == country_code][['date', 'sentiment_index']]
    df_final = pd.merge(df_final, sent_c, on='date', how='left')
    
    # 合并列
    if 'sentiment_index_y' in df_final.columns:
        df_final['sentiment_index'] = df_final['sentiment_index_y'].combine_first(df_final['sentiment_index_x'])
    
    # 插值填补情感空洞
    df_final['sentiment_index'] = df_final['sentiment_index'].interpolate()
    
    # 6. 计算残差
    df_final[true_col] = df_final[true_col].interpolate()
    df_final['residual'] = df_final[true_col] - df_final['baseline']
    
    return df_final.dropna(subset=['residual', 'sentiment_index'])

# 构建数据集
print("Building China dataset...")
df_chn = build_analysis_df(df_pred_chn, df_true_chn, 'pCN_weighted', 'national_ili_weighted', 'CHN')
print(f"China data size: {len(df_chn)}")

print("Building USA dataset...")
df_usa = build_analysis_df(df_pred_usa, df_true_usa, 'yhat', 'num_inc', 'USA')
print(f"USA data size: {len(df_usa)}")

# ================= Step 1: 修正实验 (Correction) =================

def run_correction_experiment(df, true_col, country_name):
    X = df[['sentiment_index']]
    y = df['residual']
    
    model = LinearRegression()
    model.fit(X, y)
    
    correction = model.predict(X)
    df['pred_corrected'] = df['baseline'] + correction
    
    # 评估 2024-2025
    mask_2024 = df['date'] >= '2024-01-01'
    df_eval = df[mask_2024]
    
    if len(df_eval) > 0:
        rmse_base = np.sqrt(mean_squared_error(df_eval[true_col], df_eval['baseline']))
        rmse_corr = np.sqrt(mean_squared_error(df_eval[true_col], df_eval['pred_corrected']))
        imp = (rmse_base - rmse_corr) / rmse_base * 100
        
        print(f"\n>>> {country_name} Correction Results (2024-2025) <<<")
        print(f"  Baseline RMSE: {rmse_base:,.0f}")
        print(f"  Corrected RMSE: {rmse_corr:,.0f}")
        print(f"  🚀 Improvement: {imp:.2f}%")
        print(f"  Coefficient: {model.coef_[0]:.2f} (Sensitivity)")
    else:
        print(f"\n>>> {country_name}: No data for 2024-2025 evaluation")
    
    return model, df

model_chn, df_chn_res = run_correction_experiment(df_chn, 'national_ili_weighted', 'China')
model_usa, df_usa_res = run_correction_experiment(df_usa, 'num_inc', 'USA')

# ================= Step 2: 反事实推演 (Counterfactuals) =================

def plot_counterfactual(df, true_col, model, country_name, filename):
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. 现实 (Reality)
    ax.plot(df['date'], df[true_col], color='black', linewidth=2.5, label='Observed Reality', alpha=0.8)
    
    # 2. 基准 (Baseline)
    ax.plot(df['date'], df['baseline'], color='gray', linestyle=':', label='Volume-Based Baseline', linewidth=1.5)
    
    # 3. 反事实 (Counterfactual): "Rational Society" (Sentiment = 0)
    # y = baseline + (coef * sentiment + intercept)
    # If sentiment = 0, y = baseline + intercept
    neutral_correction = model.intercept_
    df['pred_counterfactual'] = df['baseline'] + neutral_correction
    
    mask = df['date'] >= '2023-01-01'
    # 画反事实曲线
    ax.plot(df.loc[mask, 'date'], df.loc[mask, 'pred_counterfactual'], 
            color='#1F77B4', linestyle='--', linewidth=2.5, 
            label='Counterfactual: "Rational Society" (No Panic)')
    
    # 4. 填充 "Cost of Panic" (现实 > 反事实)
    # 只有当现实比“理性状态”更糟糕（病例更多）时，才是恐慌的代价
    ax.fill_between(df.loc[mask, 'date'], df.loc[mask, true_col], df.loc[mask, 'pred_counterfactual'],
                    where=(df.loc[mask, true_col] > df.loc[mask, 'pred_counterfactual']),
                    color='red', alpha=0.2, label='Excess Burden due to Panic')
    
    ax.set_title(f'Counterfactual Simulation: {country_name}\nWhat if public sentiment remained neutral?', loc='left', fontsize=14, fontweight='bold')
    ax.set_ylabel('Epidemic Intensity')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved: {filename}")

plot_counterfactual(df_usa_res, 'num_inc', model_usa, 'USA', './result/Simulation_USA_Counterfactual.png')
plot_counterfactual(df_chn_res, 'national_ili_weighted', model_chn, 'China', './result/Simulation_China_Counterfactual.png')