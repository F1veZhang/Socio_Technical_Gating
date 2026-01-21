import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    try:
        df_p_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_t_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        infra_df = pd.read_csv('./data/cleaned_digital_infrastructure.csv') # 基建数据
        
        # 统一时间
        for df in [df_p_chn, df_t_chn, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
            
        return df_p_chn, df_t_chn, sent_df, infra_df
    except FileNotFoundError:
        print("❌ 数据文件缺失")
        return None

def run_sensitivity_test(df_p, df_t, sent_df, infra_df):
    # 1. 准备全量数据
    # 计算 Residual
    df_p['raw'] = df_p['pCN'] if 'pCN' in df_p.columns else df_p['pS']*0.596 + df_p['pN']*0.404
    df_agg = df_p.groupby('date')['raw'].mean().reset_index()
    merged = pd.merge(df_agg, df_t[['date', 'national_ili_weighted']], on='date').dropna()
    mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
    mu_t, sigma_t = merged['national_ili_weighted'].mean(), merged['national_ili_weighted'].std()
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    
    df_final = pd.merge(df_t[['date', 'national_ili_weighted']], df_agg[['date', 'baseline']], on='date', how='left')
    df_final.rename(columns={'national_ili_weighted': 'truth'}, inplace=True)
    df_final['Residual'] = df_final['truth'] - df_final['baseline']
    
    # 匹配情感
    sent_c = sent_df[sent_df['country'] == 'CHN'][['date', 'sentiment_index']].copy()
    sent_c.rename(columns={'sentiment_index': 'sentiment'}, inplace=True)
    df_model = pd.merge_asof(df_final.sort_values('date'), sent_c.sort_values('date'), 
                             on='date', direction='nearest', tolerance=pd.Timedelta(days=7)).dropna()
    
    df_model['Panic'] = -1 * df_model['sentiment']
    df_model['Year'] = df_model['date'].dt.year
    
    # 2. 匹配基建数据 (按年份)
    df_model = pd.merge(df_model, infra_df[['Year', 'CNNIC_China_Internet']], on='Year', how='left')
    
    # 3. 滑动阈值测试 (50% -> 80%)
    thresholds = np.arange(50, 81, 2) # 每隔2%测一次
    results = []
    
    print("Running Sensitivity Analysis...")
    for t in thresholds:
        # 分组
        group_low = df_model[df_model['CNNIC_China_Internet'] < t]
        group_high = df_model[df_model['CNNIC_China_Internet'] >= t]
        
        # 计算相关性 (Panic vs Residual)
        # 我们关注的是：High 组的相关性是否显著高于 Low 组
        corr_low = group_low['Panic'].corr(group_low['Residual']) if len(group_low) > 10 else 0
        corr_high = group_high['Panic'].corr(group_high['Residual']) if len(group_high) > 10 else 0
        
        # 记录 Gap (提升幅度)
        gap = corr_high - corr_low
        
        results.append({
            'Threshold': t,
            'Corr_Low': corr_low,
            'Corr_High': corr_high,
            'Gap': gap,
            'N_High': len(group_high)
        })
        
    res_df = pd.DataFrame(results)
    
    # 4. 绘图
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制 High 和 Low 的相关性曲线
    ax1.plot(res_df['Threshold'], res_df['Corr_High'], 'o-', color='#D62728', linewidth=2, label='Correlation (Above Threshold)')
    ax1.plot(res_df['Threshold'], res_df['Corr_Low'], 's--', color='gray', linewidth=1.5, alpha=0.7, label='Correlation (Below Threshold)')
    
    # 绘制 Gap (提升幅度)
    ax2 = ax1.twinx()
    ax2.bar(res_df['Threshold'], res_df['Gap'], color='#1f77b4', alpha=0.2, width=1.5, label='Improvement Gap')
    
    # 找到 Gap 最大的点 (最佳阈值)
    best_t = res_df.loc[res_df['Gap'].idxmax()]['Threshold']
    
    ax1.axvline(best_t, color='black', linestyle=':', label=f'Optimal Threshold (~{best_t}%)')
    
    ax1.set_xlabel('Internet Penetration Threshold (%)')
    ax1.set_ylabel('Pearson Correlation (r)')
    ax2.set_ylabel('Performance Gap (High - Low)')
    
    ax1.set_title(f'Sensitivity Analysis: Why {best_t}%? \n(Finding the Digital Tipping Point)', fontweight='bold')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('./result/Experiment_Threshold_Sensitivity.png', dpi=300)
    print(f"✅ 敏感性分析完成。最佳统计阈值出现在: {best_t}%")
    print("结果图已保存至: Experiment_Threshold_Sensitivity.png")

if __name__ == "__main__":
    data = load_data()
    if data:
        run_sensitivity_test(*data)