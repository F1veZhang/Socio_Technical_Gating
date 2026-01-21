import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import grangercausalitytests

# ================= 配置参数 =================
ROLLING_WINDOW = 12       # 12周滚动窗口 (约3个月)
PHASE_1_END = '2020-01-23' # 疫情开始 (武汉封城)
PHASE_2_END = '2022-12-07' # 疫情结束 (新十条/放开)

# ================= 1. 数据加载 =================
def load_data():
    print("Step 1: Loading datasets...")
    try:
        # 疫情数据
        df_p_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        df_t_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        df_p_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_t_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        
        # 情感数据
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        
        # 数字基础设施数据
        infra_df = pd.read_csv('./data/cleaned_digital_infrastructure.csv')
        
        # 统一时间格式
        for df in [df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
            
        print("✅ Data loaded successfully.")
        return df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df, infra_df
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e.filename}")
        exit()

# ================= 2. 数据处理与基准构建 =================
def prep_country_data(df_pred, df_true, raw_col, true_col, sent_df, infra_df, country):
    print(f"Processing data for {country}...")
    
    # 1. 构建 Baseline (均值方差匹配)
    if raw_col == 'pCN_weighted':
        df_pred['raw'] = df_pred['pCN'] if 'pCN' in df_pred.columns else df_pred['pS']*0.596 + df_pred['pN']*0.404
    else:
        df_pred['raw'] = df_pred[raw_col]
        
    df_agg = df_pred.groupby('date')['raw'].mean().reset_index()
    merged = pd.merge(df_agg, df_true[['date', true_col]], on='date').dropna()
    
    if len(merged) > 10:
        mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
        mu_t, sigma_t = merged[true_col].mean(), merged[true_col].std()
    else:
        mu_p, sigma_p = df_agg['raw'].mean(), df_agg['raw'].std()
        mu_t, sigma_t = df_true[true_col].mean(), df_true[true_col].std()
        
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    df_agg['baseline'] = df_agg['baseline'].apply(lambda x: max(0, x))
    
    # 2. 外推填充 (Extrapolation) - 覆盖 2024-2025
    df_final = pd.merge(df_true[['date', true_col]], df_agg[['date', 'baseline']], on='date', how='left')
    df_final.rename(columns={true_col: 'truth'}, inplace=True)
    
    df_final = df_final.set_index('date')
    train_series = df_final['baseline'].dropna()
    future_idx = df_final.index[df_final.index > train_series.index.max()]
    
    if len(future_idx) > 0 and len(train_series) >= 52:
        pattern = train_series.iloc[-52:].values
        tiles = int(np.ceil(len(future_idx)/52))
        fill_values = np.tile(pattern, tiles)[:len(future_idx)]
        df_final.loc[future_idx, 'baseline'] = fill_values
    
    df_final['baseline'] = df_final['baseline'].interpolate()
    df_final = df_final.reset_index()
    
    # 3. 匹配情感数据
    sent_c = sent_df[sent_df['country'] == country][['date', 'sentiment_index']].copy()
    sent_c.rename(columns={'sentiment_index': 'sentiment'}, inplace=True)
    df = pd.merge(df_final, sent_c, on='date', how='left')
    df['sentiment'] = df['sentiment'].interpolate()
    
    # 4. 匹配基础设施数据
    infra_col = 'CNNIC_China_Internet' if country == 'CHN' else 'Pew_USA_Smartphone'
    infra_clean = infra_df[['Year', infra_col]].dropna().sort_values('Year')

    infra_clean['date'] = pd.to_datetime(infra_clean['Year'].astype(str) + '-07-01')
    
    df = pd.merge_asof(df.sort_values('date'), infra_clean[['date', infra_col]], 
                       on='date', direction='nearest', tolerance=pd.Timedelta(days=365))
    df[infra_col] = df[infra_col].interpolate(limit_direction='both')
    
    # 5. 计算核心指标
    df = df.dropna(subset=['truth', 'baseline', 'sentiment']).copy()
    df['Panic'] = -1 * df['sentiment']  # 恐慌指数 (Sentiment越负 Panic越大)
    df['Residual'] = df['truth'] - df['baseline'] # 预测偏差 (Truth > Baseline = Underestimation)
    
    # 滚动相关性
    df['Rolling_Corr'] = df['Panic'].rolling(ROLLING_WINDOW).corr(df['Residual'])
    
    return df, infra_col

# ================= 3. 统计检验 (分阶段 Granger) =================
def run_phased_granger(df, country):
    print(f"\n>>> {country} Phased Granger Causality Test <<<")
    
    phases = {
        'Phase I (Pre-COVID)': df[df['date'] < PHASE_1_END],
        'Phase II (Pandemic)': df[(df['date'] >= PHASE_1_END) & (df['date'] < PHASE_2_END)],
        'Phase III (Post-COVID)': df[df['date'] >= PHASE_2_END]
    }
    
    results_text = []
    
    for name, data in phases.items():
        if len(data) < 20: 
            continue
            
        # 检验: Panic -> Residual (Lag 1-4)
        test_data = data[['Residual', 'Panic']].dropna()
        try:
            # verbose=False 禁止打印，只获取返回值
            gc = grangercausalitytests(test_data, maxlag=4, verbose=False)
            
            # 获取最小 P 值 (最显著的滞后阶数)
            p_values = [gc[lag][0]['ssr_ftest'][1] for lag in range(1, 5)]
            min_p = min(p_values)
            best_lag = p_values.index(min_p) + 1
            
            sig_mark = "✅ Significant" if min_p < 0.05 else "❌ Not Sig"
            res_str = f"  {name:<22}: Min P-val={min_p:.4f} (Lag {best_lag}) {sig_mark}"
            print(res_str)
            
            # 为绘图准备标注文本
            if min_p < 0.05:
                results_text.append(f"{name.split('(')[0].strip()}:\nLagged Causal (p<{min_p:.3f})")
            else:
                # 额外检查同期相关性 (Instantaneous Coupling)
                corr_lag0 = data['Panic'].corr(data['Residual'])
                if corr_lag0 > 0.3:
                    results_text.append(f"{name.split('(')[0].strip()}:\nInstant Causal (r={corr_lag0:.2f})")
                else:
                    results_text.append(f"{name.split('(')[0].strip()}:\nDecoupled")
                    
        except Exception as e:
            print(f"  {name:<22}: Test Failed ({str(e)})")
            
    return results_text

# ================= 4. 绘图 (双轴 + 背景分段) =================
def plot_dual_axis(df, infra_col, country, granger_text, ax):
    # 1. 绘制左轴：滚动相关性
    color_corr = '#D62728' if country == 'CHN' else '#1f77b4' # 红(CN) / 蓝(US)
    
    ln1 = ax.plot(df['date'], df['Rolling_Corr'], color=color_corr, linewidth=1.5, label='Panic-Error Coupling')
    
    # 填充强相关区 (Crisis Coupling)
    ax.fill_between(df['date'], 0, df['Rolling_Corr'], 
                    where=(df['Rolling_Corr'] >= 0.3), color=color_corr, alpha=0.3, label='Crisis Mode (>0.3)')
    
    ax.set_ylabel('Coupling Strength (Correlation)', fontweight='bold', color=color_corr)
    ax.tick_params(axis='y', labelcolor=color_corr)
    ax.set_ylim(-0.8, 1.0)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 2. 绘制右轴：数字基础设施
    ax2 = ax.twinx()
    color_infra = 'gray'
    label_infra = 'Internet Penetration (%)' if country == 'CHN' else 'Smartphone Penetration (%)'
    
    ln2 = ax2.plot(df['date'], df[infra_col], color=color_infra, linestyle='--', linewidth=2, alpha=0.6, label=label_infra)
    
    ax2.set_ylabel(label_infra, fontweight='bold', color='#555555')
    ax2.tick_params(axis='y', labelcolor='#555555')
    
    # 统一 Y 轴范围 (40% - 100%) 方便对比
    ax2.set_ylim(40, 105)
    
    # 3. 绘制背景分段
    # Phase 1
    ax.axvspan(df['date'].min(), pd.to_datetime(PHASE_1_END), color='gray', alpha=0.05)
    # Phase 2
    ax.axvspan(pd.to_datetime(PHASE_1_END), pd.to_datetime(PHASE_2_END), color='yellow', alpha=0.05)
    # Phase 3
    ax.axvspan(pd.to_datetime(PHASE_2_END), df['date'].max(), color='green', alpha=0.05)
    
    # 4. 添加文字标注 (Granger 结果)
    # 在每个阶段的中间位置添加文字
    dates = [pd.to_datetime('2017-06-01'), pd.to_datetime('2021-06-01'), pd.to_datetime('2024-03-01')]
    for i, txt in enumerate(granger_text):
        if i < len(dates):
            ax.text(dates[i], 0.85, txt, ha='center', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 标题
    title = f"{country}: Socio-Technical Evolution (2015-2025)"
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold')
    
    # 图例合并
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower left', fontsize=9)

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 加载
    df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df, infra_df = load_data()
    
    # 2. 处理
    data_usa, col_usa = prep_country_data(df_p_usa, df_t_usa, 'yhat', 'num_inc', sent_df, infra_df, 'USA')
    data_chn, col_chn = prep_country_data(df_p_chn, df_t_chn, 'pCN_weighted', 'national_ili_weighted', sent_df, infra_df, 'CHN')
    
    # 3. 统计检验
    res_usa = run_phased_granger(data_usa, 'USA')
    res_chn = run_phased_granger(data_chn, 'China')
    
    # 4. 绘图
    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    plot_dual_axis(data_usa, col_usa, 'USA', res_usa, ax1)
    plot_dual_axis(data_chn, col_chn, 'China', res_chn, ax2)
    
    # 调整
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.set_xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2025-06-01'))
    
    plt.tight_layout()
    plt.savefig('./result/Analysis_SocioTechnical_Coupling_Full.png', dpi=300)
    print("\n✅ Analysis Complete. Plot saved to Analysis_SocioTechnical_Coupling_Full.png")