import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ================= 配置 =================
THRESHOLD_RATE = 65.0  # 您的假设阈值
PHASE_1_END = '2020-01-23' # 疫情开始
PHASE_2_END = '2022-12-07' # 疫情结束

def load_and_merge_data():
    try:
        # 加载所有数据
        df_p_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        df_t_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        df_p_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_t_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        infra_df = pd.read_csv('./data/cleaned_digital_infrastructure.csv') # 使用您刚才清洗的
        
        # 统一时间
        for df in [df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
            
        return df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df, infra_df
    except FileNotFoundError:
        print("❌ 请确保所有数据文件（包括 cleaned_digital_infrastructure.csv）都在当前目录下")
        exit()

def prep_analysis_data(df_pred, df_true, raw_col, true_col, sent_df, infra_df, country):
    # 1. 基础数据构建 (Baseline & Residual)
    if raw_col == 'pCN_weighted':
        df_pred['raw'] = df_pred['pCN'] if 'pCN' in df_pred.columns else df_pred['pS']*0.596 + df_pred['pN']*0.404
    else:
        df_pred['raw'] = df_pred[raw_col]
    
    df_agg = df_pred.groupby('date')['raw'].mean().reset_index()
    merged = pd.merge(df_agg, df_true[['date', true_col]], on='date').dropna()
    
    # Rescaling
    mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
    mu_t, sigma_t = merged[true_col].mean(), merged[true_col].std()
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    df_agg['baseline'] = df_agg['baseline'].apply(lambda x: max(0, x))
    
    df_final = pd.merge(df_true[['date', true_col]], df_agg[['date', 'baseline']], on='date', how='left')
    df_final.rename(columns={true_col: 'truth'}, inplace=True)
    df_final = df_final.dropna() # 简单处理，不做外推以保证真实性
    
    # 2. 匹配情感
    sent_c = sent_df[sent_df['country'] == country][['date', 'sentiment_index']].copy()
    sent_c.rename(columns={'sentiment_index': 'sentiment'}, inplace=True)
    df = pd.merge(df_final, sent_c, on='date', how='left')
    
    # 3. 匹配基建数据
    # 使用 CNNIC (CN) 或 Pew Smartphone (USA)
    infra_col = 'CNNIC_China_Internet' if country == 'CHN' else 'Pew_USA_Smartphone'
    infra_clean = infra_df[['Year', infra_col]].dropna().sort_values('Year')
    infra_clean['date'] = pd.to_datetime(infra_clean['Year'].astype(str) + '-07-01')
    
    df = pd.merge_asof(df.sort_values('date'), infra_clean[['date', infra_col]], 
                       on='date', direction='nearest', tolerance=pd.Timedelta(days=365))
    df['infra_rate'] = df[infra_col].interpolate(limit_direction='both')
    
    # 4. 计算变量
    df = df.dropna(subset=['truth', 'baseline', 'sentiment', 'infra_rate']).copy()
    df['Panic'] = -1 * df['sentiment']
    df['Residual'] = df['truth'] - df['baseline']
    
    # 计算局部相关性 (作为检验指标)
    # 使用较短的窗口 (8周) 来捕捉瞬间关系
    df['Local_Corr'] = df['Panic'].rolling(8).corr(df['Residual'])
    
    # 标记疫情期
    df['Is_Pandemic'] = (df['date'] >= PHASE_1_END) & (df['date'] < PHASE_2_END)
    
    return df

def run_threshold_experiment(df_usa, df_chn):
    # 合并两国数据进行整体检验
    df_usa['Country'] = 'USA'
    df_chn['Country'] = 'China'
    df_all = pd.concat([df_usa, df_chn])
    
    # === 分组逻辑 ===
    def assign_group(row):
        if row['Is_Pandemic']:
            return 'Group C: Pandemic\n(Disrupted)'
        elif row['infra_rate'] < THRESHOLD_RATE:
            return 'Group A: Low Infra\n(<65%)'
        else:
            return 'Group B: High Infra\n(>65%)' # 这是我们要验证的强相关组
            
    df_all['Exp_Group'] = df_all.apply(assign_group, axis=1)
    
    # 去除空值 (由于 Rolling 产生的 NaN)
    df_clean = df_all.dropna(subset=['Local_Corr'])
    
    # === 统计检验 ===
    print(f"\n>>> 阈值验证实验 (Threshold = {THRESHOLD_RATE}%) <<<")
    groups = ['Group A: Low Infra\n(<65%)', 'Group B: High Infra\n(>65%)', 'Group C: Pandemic\n(Disrupted)']
    
    stats_res = []
    for g in groups:
        data = df_clean[df_clean['Exp_Group'] == g]['Local_Corr']
        mean_corr = data.mean()
        median_corr = data.median()
        # 计算正相关比例 (Correlation > 0)
        pos_ratio = (data > 0).mean() * 100
        stats_res.append({'Group': g, 'Mean_Corr': mean_corr, 'Pos_Ratio': pos_ratio, 'N': len(data)})
        print(f"{g.splitlines()[0]:<20}: Avg Corr = {mean_corr:.3f} | Positive% = {pos_ratio:.1f}% | N={len(data)}")

    # T-test: Group B (High) vs Group A (Low)
    g_high = df_clean[df_clean['Exp_Group'] == groups[1]]['Local_Corr']
    g_low = df_clean[df_clean['Exp_Group'] == groups[0]]['Local_Corr']
    
    t_stat, p_val = stats.ttest_ind(g_high, g_low, nan_policy='omit')
    print("\n>>> 假设检验 (Hypothesis Test) <<<")
    print(f"H0: High Infra Group == Low Infra Group")
    print(f"Result: T-stat={t_stat:.3f}, P-value={p_val:.4e}")
    if p_val < 0.05 and t_stat > 0:
        print("✅ 验证成功！高普及率组的相关性显著高于低普及率组。")
    else:
        print("⚠️ 验证不显著，可能阈值设置有误或存在其他变量。")

    # === 绘图 ===
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 箱线图
    sns.boxplot(x='Exp_Group', y='Local_Corr', data=df_clean, order=groups, ax=ax, palette=['gray', '#D62728', '#FBC02D'])
    
    ax.set_title(f'Verification Experiment: Does >{THRESHOLD_RATE}% Penetration Amplify Coupling?', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coupling Strength (8-week Rolling Correlation)', fontsize=12)
    ax.set_xlabel('Experiment Conditions', fontsize=12)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('./result/Experiment_Threshold_Verification.png', dpi=300)
    print("✅ 实验结果图已保存: Experiment_Threshold_Verification.png")

if __name__ == "__main__":
    df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df, infra_df = load_and_merge_data()
    
    data_usa = prep_analysis_data(df_p_usa, df_t_usa, 'yhat', 'num_inc', sent_df, infra_df, 'USA')
    data_chn = prep_analysis_data(df_p_chn, df_t_chn, 'pCN_weighted', 'national_ili_weighted', sent_df, infra_df, 'CHN')
    
    run_threshold_experiment(data_usa, data_chn)