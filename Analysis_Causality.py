import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR

# ================= 配置 =================
START_DATE_POST_COVID = '2023-01-01'
MAX_LAG = 4  # 测试最大滞后4周

# ================= 1. 数据加载与预处理 =================
def load_and_prep():
    print("正在加载数据...")
    try:
        df_pred_usa = pd.read_csv('./data/pred_detail_USA_weekly.csv')
        df_true_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')
        df_pred_chn = pd.read_csv('./data/pred_detail_CHN_week.csv')
        df_true_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
        sent_df = pd.read_csv('./data/weekly_sentiment_series_FINAL.csv')
        
        for df in [df_pred_usa, df_true_usa, df_pred_chn, df_true_chn, sent_df]:
            d_col = 'date' if 'date' in df.columns else 'week_start'
            df['date'] = pd.to_datetime(df[d_col])
        return df_pred_usa, df_true_usa, df_pred_chn, df_true_chn, sent_df
    except FileNotFoundError:
        print("❌ 缺少文件")
        exit()

def get_residuals(df_pred, df_true, raw_col, true_col, sent_df, country):
    # 构建 Baseline
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
    
    # Merge Truth & Sentiment
    df = pd.merge(df_true, df_agg[['date', 'baseline']], on='date', how='left')
    df.rename(columns={true_col: 'truth'}, inplace=True)
    
    sent_c = sent_df[sent_df['country'] == country][['date', 'sentiment_index']].copy()
    sent_c.columns = ['date', 'sentiment']
    df = pd.merge(df, sent_c, on='date', how='left')
    
    # Filter Post-COVID
    mask = df['date'] >= START_DATE_POST_COVID
    df_post = df.loc[mask].dropna(subset=['truth', 'baseline', 'sentiment']).copy()
    
    # 计算核心变量：Sentiment 和 Residual
    # 注意：为了符合直觉（恐慌导致病例），我们将 Sentiment 取负，变成 "Panic Index"
    # 这样 Sentiment 升高(越恐慌) -> Residual 升高(低估越多)，正相关更易读
    df_post['Panic'] = -1 * df_post['sentiment'] 
    df_post['Residual'] = df_post['truth'] - df_post['baseline']
    
    return df_post[['date', 'Panic', 'Residual']].set_index('date').sort_index()

# ================= 2. 格兰杰因果检验 =================
def run_granger(df, country):
    print(f"\n>>> {country} 格兰杰因果检验 (Granger Causality) <<<")
    # 检验: Panic 是否导致 Residual?
    # 输入格式: [Response(Y), Predictor(X)] -> [Residual, Panic]
    data = df[['Residual', 'Panic']]
    
    gc_res = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)
    
    print(f"{'Lag':<5} {'F-test':<10} {'p-value':<10} {'Result'}")
    print("-" * 40)
    
    significant_lags = []
    for lag in range(1, MAX_LAG + 1):
        test_result = gc_res[lag][0]['ssr_ftest']
        f_val = test_result[0]
        p_val = test_result[1]
        is_sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"{lag:<5} {f_val:<10.4f} {p_val:<10.4f} {is_sig}")
        if p_val < 0.05: significant_lags.append(lag)
        
    if significant_lags:
        print(f"✅ 结论: 在滞后 {significant_lags} 周时，恐慌情绪显著驱动了疫情偏差！")
    else:
        print("⚠️ 结论: 未检测到显著的线性因果关系 (可能存在非线性或数据噪音)。")

# ================= 3. 脉冲响应分析 (Impulse Response) =================
def run_impulse_response(df, country, filename):
    # 建立向量自回归模型 (VAR)
    model = VAR(df)
    # 自动选择最佳滞后阶数 (AIC准则)
    results = model.fit(maxlags=MAX_LAG, ic='aic')
    
    print(f"\n>>> {country} 脉冲响应分析 (VAR Model) <<<")
    print(f"最佳滞后阶数 (Best Lag): {results.k_ar}")
    
    # 计算脉冲响应 (10周)
    irf = results.irf(10)
    
    # 绘图
    plt.style.use('seaborn-v0_8-white')
    # 我们只关心 Panic -> Residual 的影响
    # orth=False 表示非正交化冲击，适合这种弱耦合系统
    fig = irf.plot(impulse='Panic', response='Residual', orth=False, figsize=(10, 6))
    
    # 调整图表标题和样式
    plt.suptitle(f'Impulse Response: Impact of "Panic Shock" on "Epidemic Deviation"\n({country}, Post-COVID)', fontsize=14, fontweight='bold', y=1.02)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # 保存
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ 脉冲响应图已保存: {filename}")

# ================= 主程序 =================
if __name__ == "__main__":
    df_p_usa, df_t_usa, df_p_chn, df_t_chn, sent_df = load_and_prep()
    
    # 准备数据 (Post-COVID)
    data_usa = get_residuals(df_p_usa, df_t_usa, 'yhat', 'num_inc', sent_df, 'USA')
    data_chn = get_residuals(df_p_chn, df_t_chn, 'pCN_weighted', 'national_ili_weighted', sent_df, 'CHN')
    
    # 1. 运行 USA 分析
    run_granger(data_usa, 'USA')
    run_impulse_response(data_usa, 'USA', './result/Analysis_Impulse_Response_USA.png')
    
    # 2. 运行 China 分析
    run_granger(data_chn, 'China')
    run_impulse_response(data_chn, 'China', './result/Analysis_Impulse_Response_China.png')