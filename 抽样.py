import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置 =================
# 1. 强力锁定随机种子 (确保每次洗牌都一样)
np.random.seed(42)

START_DATE = '2019-01-01'  # 锁定 2020-2025
PLACEBO_ROUNDS = 2000      # 增加抽样次数 (让分布更稳定)
TEST_ITERS = 5             # 连续测试 5 次，看结果是否抖动

def load_data():
    try:
        df_p_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        df_t_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        
        for df in [df_p_usa, df_t_usa, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
        return df_p_usa, df_t_usa, sent_df
    except:
        return None

def prep_usa_data(df_p, df_t, sent_df):
    # 构建 Baseline
    df_p['raw'] = df_p['yhat']
    df_agg = df_p.groupby('date')['raw'].mean().reset_index()
    merged = pd.merge(df_agg, df_t[['date', 'num_inc']], on='date').dropna()
    
    mu_p, sigma_p = merged['raw'].mean(), merged['raw'].std()
    mu_t, sigma_t = merged['num_inc'].mean(), merged['num_inc'].std()
    
    df_agg['baseline'] = (df_agg['raw'] - mu_p) / sigma_p * sigma_t + mu_t
    df_final = pd.merge(df_t[['date', 'num_inc']], df_agg[['date', 'baseline']], on='date', how='left')
    df_final.rename(columns={'num_inc': 'truth'}, inplace=True)
    
    # 匹配情感
    sent_c = sent_df[sent_df['country'] == 'USA'][['date', 'sentiment_index']].copy()
    sent_c.rename(columns={'sentiment_index': 'sentiment'}, inplace=True)
    
    df = pd.merge_asof(df_final.sort_values('date'), sent_c.sort_values('date'), 
                       on='date', direction='nearest', tolerance=pd.Timedelta(days=7)).dropna()
    
    # 锁定时间窗口
    df = df[df['date'] >= START_DATE].copy()
    
    df['Panic'] = -1 * df['sentiment']
    df['Residual'] = df['truth'] - df['baseline']
    
    return df['Panic'].values, df['Residual'].values

def run_single_audit(x, y, round_id):
    # 1. 计算 True Correlation (这个绝对不能变)
    true_r = np.corrcoef(x, y)[0, 1]
    
    # 2. 运行安慰剂抽样
    null_dist = []
    y_shuffled = y.copy()
    
    # 注意：虽然我们在开头锁了种子，但如果在循环里不重置，
    # 连续跑5次的结果会因为random state的推进而不同。
    # 为了验证“代码稳定性”，我们在每次 Audit 内部都重置种子，看看能不能复现一模一样的结果。
    rng = np.random.default_rng(42) # 局部锁定
    
    for _ in range(PLACEBO_ROUNDS):
        rng.shuffle(y_shuffled)
        null_dist.append(np.corrcoef(x, y_shuffled)[0, 1])
    
    null_dist = np.array(null_dist)
    
    # 3. 计算 P 值
    if true_r > 0:
        p_val = (null_dist >= true_r).mean()
    else:
        p_val = (null_dist <= true_r).mean()
        
    print(f"  [Round {round_id}] True r: {true_r:.6f} | P-value: {p_val:.6f} | Null Mean: {null_dist.mean():.6f}")
    return true_r, p_val, null_dist

if __name__ == "__main__":
    data = load_data()
    if data:
        df_p_usa, df_t_usa, sent_df = data
        x, y = prep_usa_data(df_p_usa, df_t_usa, sent_df)
        
        print(f"数据点数量: {len(x)}")
        print(f"正在进行 {TEST_ITERS} 次重复审计 (应完全一致)...")
        
        results = []
        last_null_dist = None
        
        for i in range(1, TEST_ITERS + 1):
            res = run_single_audit(x, y, i)
            results.append(res)
            last_null_dist = res[2]
            
        # 绘图检查
        true_r = results[0][0]
        plt.figure(figsize=(10, 6))
        sns.histplot(last_null_dist, bins=50, kde=True, color='lightgray', label='Null Distribution (Random)')
        plt.axvline(true_r, color='blue', linewidth=2, linestyle='--', label=f'True Correlation (r={true_r:.3f})')
        
        plt.title('Audit: USA Placebo Test Stability (Fixed Seed)', fontweight='bold')
        plt.xlabel('Correlation Coefficient')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./result/Debug_USA_Placebo.png')
        print("\n✅ 审计图已保存: Debug_USA_Placebo.png")