import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置 =================
# 定义冬季月份 (10月 - 次年3月)
WINTER_MONTHS = [10, 11, 12, 1, 2, 3]

def load_data():
    try:
        df_t_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        df_p_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        
        # 统一时间格式
        for df in [df_t_usa, df_p_usa, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
            
        return df_t_usa, df_p_usa, sent_df
    except FileNotFoundError:
        print("❌ 缺少数据文件 (aligned_data_usa_complete.csv 等)")
        return None, None, None

def prep_usa_data(df_t, df_p, sent_df):
    # 1. 构建 Baseline
    df_p['raw'] = df_p['yhat']
    df_agg = df_p.groupby('date')['raw'].mean().reset_index()
    merged = pd.merge(df_agg, df_t[['date', 'num_inc']], on='date').dropna()
    
    mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
    mu_t, sigma_t = merged['num_inc'].mean(), merged['num_inc'].std()
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    
    df_final = pd.merge(df_t[['date', 'num_inc']], df_agg[['date', 'baseline']], on='date', how='left')
    
    # 2. 匹配情感
    sent_usa = sent_df[sent_df['country'] == 'USA'][['date', 'sentiment_index']].copy()
    df = pd.merge(df_final, sent_usa, on='date', how='inner')
    
    # 3. 计算指标
    df['Panic'] = -1 * df['sentiment_index']
    df['Residual'] = df['num_inc'] - df['baseline']
    
    # 4. 标记季节
    df['Month'] = df['date'].dt.month
    # 确保这里的标签和下面 palette 的 key 完全一致
    df['Season'] = df['Month'].apply(lambda x: 'Winter (Oct-Mar)' if x in WINTER_MONTHS else 'Non-Winter (Apr-Sep)')
    
    return df.dropna()

def analyze_seasonality(df):
    print(">>> 美国数据季节性分组分析 <<<")
    
    # 分组计算相关性
    groups = {
        'Overall (All Time)': df,
        'Winter (Flu Season)': df[df['Season'] == 'Winter (Oct-Mar)'],
        'Non-Winter (Off Season)': df[df['Season'] == 'Non-Winter (Apr-Sep)']
    }
    
    results = []
    for name, data in groups.items():
        if len(data) > 0:
            corr = data['Panic'].corr(data['Residual'])
            results.append({'Group': name, 'Correlation': corr, 'N': len(data)})
            print(f"{name:<25}: r = {corr:.4f} (N={len(data)})")
            
    res_df = pd.DataFrame(results)
    
    # === 绘图 ===
    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 散点图对比 (修复了 palette 的 key)
    sns.scatterplot(data=df, x='Panic', y='Residual', hue='Season', style='Season', 
                    palette={'Winter (Oct-Mar)': '#D62728', 'Non-Winter (Apr-Sep)': '#1f77b4'}, 
                    alpha=0.7, ax=ax1)
    
    # 添加回归线
    sns.regplot(data=df[df['Season'] == 'Winter (Oct-Mar)'], x='Panic', y='Residual', 
                scatter=False, color='#D62728', ax=ax1, label='Winter Trend')
    sns.regplot(data=df[df['Season'] == 'Non-Winter (Apr-Sep)'], x='Panic', y='Residual', 
                scatter=False, color='#1f77b4', ax=ax1, label='Non-Winter Trend')
    
    ax1.set_title('Scatter Plot: Panic vs. Residual by Season', fontweight='bold')
    ax1.legend()
    
    # 2. 柱状图对比相关性
    colors = ['gray', '#D62728', '#1f77b4']
    # 确保柱状图标签和 results 中的 Group 名称一致
    bars = ax2.bar(res_df['Group'], res_df['Correlation'], color=colors, alpha=0.8)
    
    # 标注数值
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom' if height>0 else 'top', fontweight='bold')
        
    ax2.set_title('Correlation Strength: Is Winter Diluting the Signal?', fontweight='bold')
    ax2.set_ylabel('Pearson Correlation (r)')
    ax2.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('./result/Analysis_USA_Seasonality_Compare.png', dpi=300)
    print("\n✅ 图表已保存: Analysis_USA_Seasonality_Compare.png")

if __name__ == "__main__":
    df_t, df_p, sent_df = load_data()
    if df_t is not None:
        data = prep_usa_data(df_t, df_p, sent_df)
        analyze_seasonality(data)