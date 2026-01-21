import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ================= 配置部分 =================
START_DATE_POST_COVID = '2023-01-01'  # 定义后疫情时代的起始点

# ================= 1. 数据加载与预处理 =================
def load_and_preprocess():
    print("正在加载数据...")
    try:
        df_pred_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_pred_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        df_true_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        df_true_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        sentiment_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        
        # 统一转换时间格式
        for df in [df_pred_chn, df_pred_usa, df_true_chn, df_true_usa, sentiment_df]:
            df['date'] = pd.to_datetime(df['date'])
            
        print("✅ 数据加载完成")
        return df_pred_chn, df_pred_usa, df_true_chn, df_true_usa, sentiment_df
    
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到文件 {e.filename}。请确保所有 CSV 都在当前目录下。")
        exit()

# ================= 2. 构建生物学基准 (Baseline Construction) =================
def build_baseline_dataset(df_pred, df_true, raw_col, true_col, country_code, sent_df):
    # 1. 提取原始预测值 (Volume-based)
    if raw_col == 'pCN_weighted':
        if 'pCN' in df_pred.columns:
            df_pred['raw'] = df_pred['pCN']
        else:
            df_pred['raw'] = df_pred['pS'] * 0.596 + df_pred['pN'] * 0.404
    else:
        df_pred['raw'] = df_pred[raw_col]
        
    # 按周聚合去重
    df_agg = df_pred.groupby('date')['raw'].mean().reset_index()
    
    # 2. Rescaling (均值方差匹配) - 解决量级不一致问题
    # 使用全量数据计算统计特征
    merged = pd.merge(df_agg, df_true[['date', true_col]], on='date').dropna()
    if len(merged) > 10:
        mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
        mu_t, sigma_t = merged[true_col].mean(), merged[true_col].std()
    else:
        mu_p, sigma_p = df_agg['raw'].mean(), df_agg['raw'].std()
        mu_t, sigma_t = df_true[true_col].mean(), df_true[true_col].std()
        
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    df_agg['baseline'] = df_agg['baseline'].apply(lambda x: max(0, x)) # 修正负值
    
    # 3. 拼接真实值
    df_final = pd.merge(df_true, df_agg[['date', 'baseline']], on='date', how='left')
    
    # 4. 填充基准线空缺 (Extrapolation) - 针对 2024-2025
    df_final = df_final.set_index('date')
    train_series = df_final['baseline'].dropna()
    full_idx = df_final.index
    future_idx = full_idx[full_idx > train_series.index.max()]
    
    # 使用最后52周模式进行克隆
    if len(future_idx) > 0 and len(train_series) >= 52:
        pattern = train_series.iloc[-52:].values
        tiles = int(np.ceil(len(future_idx)/52))
        fill_values = np.tile(pattern, tiles)[:len(future_idx)]
        df_final.loc[future_idx, 'baseline'] = fill_values
    elif len(future_idx) > 0:
        df_final.loc[future_idx, 'baseline'] = train_series.mean()
        
    df_final['baseline'] = df_final['baseline'].interpolate()
    df_final = df_final.reset_index()
    
    # 5. 匹配情感数据
    sent_country = sent_df[sent_df['country'] == country_code][['date', 'sentiment_index']]
    df_final = pd.merge(df_final, sent_country, on='date', how='left')
    
    # 合并列逻辑
    if 'sentiment_index_y' in df_final.columns:
        df_final['sentiment_index'] = df_final['sentiment_index_y'].combine_first(df_final['sentiment_index_x'])
    
    # 填补情感空缺
    df_final['sentiment_index'] = df_final['sentiment_index'].interpolate()
    
    # 6. 计算残差
    df_final[true_col] = df_final[true_col].interpolate()
    df_final['residual'] = df_final[true_col] - df_final['baseline']
    
    return df_final.dropna(subset=['residual', 'sentiment_index'])

# ================= 3. 动态修正实验 (Dynamic Correction Core) =================
def run_dynamic_correction(df, true_col, country_name):
    # 1. 特征工程：构造时滞 (Lags) 和 差分 (Diff)
    df = df.sort_values('date').copy()
    
    # 动态特征：不仅看当前，还看过去和变化速度
    df['sent_lag1'] = df['sentiment_index'].shift(1)  # 滞后1周
    df['sent_lag2'] = df['sentiment_index'].shift(2)  # 滞后2周
    df['sent_diff'] = df['sentiment_index'].diff()    # 变化率 (一阶差分)
    
    # 2. 筛选 Post-COVID 时段 (2023-2025)
    mask = (df['date'] >= START_DATE_POST_COVID)
    df_period = df.loc[mask].dropna().copy()
    
    if len(df_period) < 10:
        print(f"⚠️ {country_name}: Post-COVID 数据不足，跳过分析")
        return None
    
    y_true = df_period['residual']
    
    # --- 模型 A: 简单修正 (Static) ---
    X_simple = df_period[['sentiment_index']]
    model_simple = LinearRegression().fit(X_simple, y_true)
    df_period['pred_simple'] = df_period['baseline'] + model_simple.predict(X_simple)
    
    # --- 模型 B: 动态修正 (Dynamic) ---
    # 使用 情感 + 滞后1周 + 滞后2周
    X_dynamic = df_period[['sentiment_index', 'sent_lag1', 'sent_lag2']] 
    model_dynamic = LinearRegression().fit(X_dynamic, y_true)
    df_period['pred_dynamic'] = df_period['baseline'] + model_dynamic.predict(X_dynamic)
    
    # 3. 计算 RMSE 指标
    rmse_base = np.sqrt(mean_squared_error(df_period[true_col], df_period['baseline']))
    rmse_simple = np.sqrt(mean_squared_error(df_period[true_col], df_period['pred_simple']))
    rmse_dynamic = np.sqrt(mean_squared_error(df_period[true_col], df_period['pred_dynamic']))
    
    imp_simple = (rmse_base - rmse_simple) / rmse_base * 100
    imp_dynamic = (rmse_base - rmse_dynamic) / rmse_base * 100
    
    print(f"\n>>> {country_name} Dynamic Correction Results ({START_DATE_POST_COVID} - 2025) <<<")
    print(f"  Baseline RMSE           : {rmse_base:,.0f}")
    print(f"  Simple Correction RMSE  : {rmse_simple:,.0f} (Imp: {imp_simple:.2f}%)")
    print(f"  Dynamic Correction RMSE : {rmse_dynamic:,.0f} (Imp: {imp_dynamic:.2f}%)")
    print(f"  🎯 额外提升 (Dynamic vs Simple): +{imp_dynamic - imp_simple:.2f}%")
    
    return df_period

# ================= 4. 绘图函数 =================
def plot_correction_comparison(df, true_col, country_name, filename):
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 1. 现实 (Black)
    ax.plot(df['date'], df[true_col], color='black', linewidth=2.5, label='Observed Reality', alpha=0.8)
    
    # 2. 基准 (Gray Dashed)
    ax.plot(df['date'], df['baseline'], color='gray', linestyle='--', label='Baseline (Volume-Based)', linewidth=1.5, alpha=0.7)
    
    # 3. 简单修正 (Green Dotted)
    ax.plot(df['date'], df['pred_simple'], color='green', linestyle=':', label='Simple Correction (Static)', linewidth=2)
    
    # 4. 动态修正 (Red Solid) - 
    ax.plot(df['date'], df['pred_dynamic'], color='#D62728', linestyle='-', label='Dynamic Correction (Lags)', linewidth=2.5)
    
    ax.set_title(f'{country_name} Post-COVID Surveillance: Dynamic vs Static Correction\n(2023-2025)', loc='left', fontsize=16, fontweight='bold')
    ax.set_ylabel('Epidemic Intensity', fontsize=12)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 标注提升
    # 在图上写出具体的 RMSE 提升
    # ax.text(0.02, 0.05, 'Dynamic Correction reduces Error by XX%', transform=ax.transAxes, fontsize=12, fontweight='bold', color='#D62728')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ 图表已保存: {filename}")

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 加载
    df_p_chn, df_p_usa, df_t_chn, df_t_usa, sent_df = load_and_preprocess()
    
    # 2. 构建基准数据集
    print("\n正在构建分析数据集...")
    df_chn = build_baseline_dataset(df_p_chn, df_t_chn, 'pCN_weighted', 'national_ili_weighted', 'CHN', sent_df)
    df_usa = build_baseline_dataset(df_p_usa, df_t_usa, 'yhat', 'num_inc', 'USA', sent_df)
    
    # 3. 运行动态修正实验
    print("\n开始动态修正分析...")
    res_chn = run_dynamic_correction(df_chn, 'national_ili_weighted', 'China')
    res_usa = run_dynamic_correction(df_usa, 'num_inc', 'USA')
    
    # 4. 绘图
    if res_usa is not None:
        plot_correction_comparison(res_usa, 'num_inc', 'USA', './result/Analysis_USA_Dynamic_Correction.png')
    
    if res_chn is not None:
        plot_correction_comparison(res_chn, 'national_ili_weighted', 'China', './result/Analysis_China_Dynamic_Correction.png')
        
    print("\n全部完成！请查看生成的 Analysis_*.png 图片。")