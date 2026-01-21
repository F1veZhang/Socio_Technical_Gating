import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ================= 配置 =================
BEST_THRESHOLD = 62.0  # 使用我们新发现的最佳阈值

def load_data():
    try:
        df_p_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_t_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        infra_df = pd.read_csv('./data/cleaned_digital_infrastructure.csv') 
        
        for df in [df_p_chn, df_t_chn, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
        return df_p_chn, df_t_chn, sent_df, infra_df
    except:
        return None

def verify_best_threshold(df_p, df_t, sent_df, infra_df):
    # 数据准备 (同前)
    df_p['raw'] = df_p['pCN'] if 'pCN' in df_p.columns else df_p['pS']*0.596 + df_p['pN']*0.404
    df_agg = df_p.groupby('date')['raw'].mean().reset_index()
    merged = pd.merge(df_agg, df_t[['date', 'national_ili_weighted']], on='date').dropna()
    mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
    mu_t, sigma_t = merged['national_ili_weighted'].mean(), merged['national_ili_weighted'].std()
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    
    df_final = pd.merge(df_t[['date', 'national_ili_weighted']], df_agg[['date', 'baseline']], on='date', how='left')
    df_final['Residual'] = df_final['national_ili_weighted'] - df_final['baseline']
    
    sent_c = sent_df[sent_df['country'] == 'CHN'][['date', 'sentiment_index']].copy()
    sent_c.rename(columns={'sentiment_index': 'sentiment'}, inplace=True)
    df_model = pd.merge_asof(df_final.sort_values('date'), sent_c.sort_values('date'), 
                             on='date', direction='nearest', tolerance=pd.Timedelta(days=7)).dropna()
    
    df_model['Panic'] = -1 * df_model['sentiment']
    df_model['Year'] = df_model['date'].dt.year
    df_model = pd.merge(df_model, infra_df[['Year', 'CNNIC_China_Internet']], on='Year', how='left')

    # === 分组对比 (使用新阈值 62%) ===
    group_low = df_model[df_model['CNNIC_China_Internet'] < BEST_THRESHOLD]
    group_high = df_model[df_model['CNNIC_China_Internet'] >= BEST_THRESHOLD]
    
    corr_low = group_low['Panic'].corr(group_low['Residual'])
    corr_high = group_high['Panic'].corr(group_high['Residual'])
    
    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Below 62%\n(Pre-Threshold)', 'Above 62%\n(Post-Threshold)'], 
                  [corr_low, corr_high], 
                  color=['gray', '#D62728'], alpha=0.8, width=0.6)
    
    # 添加数值标签
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')
    
    # 装饰
    ax.set_ylabel('Correlation (Panic vs. Epidemic Error)', fontsize=12)
    ax.set_title(f'Verification of the "Critical Mass" (Threshold = {BEST_THRESHOLD}%)', fontweight='bold', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加显著性标记
    x1, x2 = 0, 1
    y, h = max(corr_low, corr_high) + 0.05, 0.02
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color='k', fontsize=15)
    
    plt.tight_layout()
    plt.savefig('./result/Experiment_Threshold_Update_62.png', dpi=300)
    print(f"\n✅ 62% 验证图已生成: Experiment_Threshold_Update_62.png")
    print(f"  Low Group (N={len(group_low)}): r = {corr_low:.4f}")
    print(f"  High Group (N={len(group_high)}): r = {corr_high:.4f}")

if __name__ == "__main__":
    data = load_data()
    if data:
        verify_best_threshold(*data)