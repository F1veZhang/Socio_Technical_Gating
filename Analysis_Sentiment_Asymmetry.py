import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置 =================
PHASE_3_START = '2015-12-07'  # 后疫情时代起点

def load_data():
    try:
        df_p_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        df_t_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        df_p_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_t_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        
        for df in [df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
            
        return df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df
    except FileNotFoundError:
        print("❌ 数据文件缺失")
        exit()

def prep_adaptive_data(df_pred, df_true, raw_col, true_col, sent_df, country):
    print(f"Processing {country}...")
    
    # 1. Baseline 构建
    if raw_col == 'pCN_weighted':
        df_pred['raw'] = df_pred['pCN'] if 'pCN' in df_pred.columns else df_pred['pS']*0.596 + df_pred['pN']*0.404
    else:
        df_pred['raw'] = df_pred[raw_col]
        
    df_agg = df_pred.groupby('date')['raw'].mean().reset_index()
    merged = pd.merge(df_agg, df_true[['date', true_col]], on='date').dropna()
    
    mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
    mu_t, sigma_t = merged[true_col].mean(), merged[true_col].std()
    
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    df_final = pd.merge(df_true[['date', true_col]], df_agg[['date', 'baseline']], on='date', how='left')
    df_final.rename(columns={true_col: 'truth'}, inplace=True)
    
    # 2. 情感合并 (修复点：使用 merge_asof 解决中国数据 NaN 问题)
    sent_c = sent_df[sent_df['country'] == country][['date', 'sentiment_index']].copy()
    sent_c.rename(columns={'sentiment_index': 'sentiment'}, inplace=True)
    sent_c = sent_c.sort_values('date')
    df_final = df_final.sort_values('date')
    
    df = pd.merge_asof(df_final, sent_c, on='date', direction='nearest', tolerance=pd.Timedelta(days=7))
    
    # 3. 筛选时间段 (Post-COVID)
    df_post = df[df['date'] >= PHASE_3_START].dropna(subset=['truth', 'baseline', 'sentiment']).copy()
    
    if len(df_post) < 10:
        print(f"⚠️ {country} 数据不足 (<10行)，无法分析。")
        return None

    # 4. 自适应阈值分割 (修复点：使用中位数解决美国数据全负问题)
    # 计算当前时间段的中位数
    threshold = df_post['sentiment'].median()
    print(f"  -> {country} Split Threshold (Median): {threshold:.3f}")
    
    # 相对正面 (Relative Positive): 高于中位数
    df_post['Rel_Pos_Force'] = df_post['sentiment'].apply(lambda x: x - threshold if x > threshold else 0)
    
    # 相对负面 (Relative Negative): 低于中位数 (取绝对值距离)
    df_post['Rel_Neg_Force'] = df_post['sentiment'].apply(lambda x: abs(x - threshold) if x < threshold else 0)
    
    df_post['Residual'] = df_post['truth'] - df_post['baseline']
    
    return df_post

def analyze_asymmetry(df, country):
    if df is None: return 0, 0
    
    print(f">>> {country} Correlation Analysis <<<")
    # 计算相关性
    # Rel_Neg_Force (Panic) -> 预期正相关 (恐慌导致误差变大)
    corr_neg = df['Rel_Neg_Force'].corr(df['Residual'])
    
    # Rel_Pos_Force (Comfort) -> 预期负相关 (信心导致误差变小) 或 无关
    corr_pos = df['Rel_Pos_Force'].corr(df['Residual'])
    
    print(f"  Panic Impact (Negative Side): r = {corr_neg:.4f}")
    print(f"  Comfort Impact (Positive Side): r = {corr_pos:.4f}")
    
    return corr_neg, corr_pos

# ================= 主程序 =================
if __name__ == "__main__":
    df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df = load_data()
    
    # 准备数据
    data_usa = prep_adaptive_data(df_p_usa, df_t_usa, 'yhat', 'num_inc', sent_df, 'USA')
    data_chn = prep_adaptive_data(df_p_chn, df_t_chn, 'pCN_weighted', 'national_ili_weighted', sent_df, 'CHN')
    
    # 分析
    r_neg_usa, r_pos_usa = analyze_asymmetry(data_usa, 'USA')
    r_neg_chn, r_pos_chn = analyze_asymmetry(data_chn, 'China')
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['USA', 'China']
    neg_vals = [r_neg_usa, r_neg_chn]
    pos_vals = [r_pos_usa, r_pos_chn]
    
    x = np.arange(len(labels))
    width = 0.35
    
    # 红色: 加剧恐慌的影响 (Worsening)
    rects1 = ax.bar(x - width/2, neg_vals, width, label='Worsening Sentiment (Relative Panic)', color='#D62728')
    # 绿色: 缓解恐慌的影响 (Improving)
    rects2 = ax.bar(x + width/2, pos_vals, width, label='Improving Sentiment (Relative Comfort)', color='#2ca02c')
    
    ax.set_ylabel('Impact on Epidemic Error (Correlation)')
    ax.set_title(f'Asymmetry Test (Adaptive Median Split)\nWhich drives the error: Panic or Comfort?', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend()
    
    ax.bar_label(rects1, fmt='%.3f')
    ax.bar_label(rects2, fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig('./result/Analysis_Sentiment_Asymmetry_Final.png', dpi=300)
    print("\n✅ 最终分析图已保存: Analysis_Sentiment_Asymmetry_Final.png")