import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# ================= 配置 =================
# 分析窗口
START_DATE = '2015-01-01'
END_DATE   = '2025-12-31'

# 路径
PATH_USA_PRED = './data/pred_detail_USA_weekly.csv'
PATH_USA_TRUE = './data/aligned_data_usa_complete.csv'
PATH_CHN_PRED = './data/pred_detail_CHN_week.csv'
PATH_CHN_TRUE = './data/aligned_data_china_complete.csv'
PATH_SENTIMENT = './data/weekly_sentiment_series_FINAL.csv'

def load_data():
    df_p_usa = pd.read_csv(PATH_USA_PRED)
    df_t_usa = pd.read_csv(PATH_USA_TRUE)
    df_p_chn = pd.read_csv(PATH_CHN_PRED)
    df_t_chn = pd.read_csv(PATH_CHN_TRUE)
    df_sent  = pd.read_csv(PATH_SENTIMENT)
    
    # 统一日期
    for df in [df_p_usa, df_t_usa, df_p_chn, df_t_chn, df_sent]:
        col = 'date' if 'date' in df.columns else 'week_start'
        df['date'] = pd.to_datetime(df[col])
    
    return df_p_usa, df_t_usa, df_p_chn, df_t_chn, df_sent

def analyze_mechanism(df_pred, df_true, df_sent, country_code, true_col, pred_col):
    # 1. 数据对齐
    df = pd.merge(df_true[['date', true_col]], df_pred[['date', pred_col]], on='date')
    sent_c = df_sent[df_sent['country'] == country_code][['date', 'sentiment_index']].copy()
    df = pd.merge(df, sent_c, on='date', how='inner')
    df = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)].copy()
    
    # 2. 计算“预测残差” (Residuals)
    # Residual = 真实就医量 - 模型预测量
    # 我们要看：Sentiment 到底能不能解释这个“意料之外的波动”？
    
    # 标准化 
    y_true_z = (df[true_col] - df[true_col].mean()) / df[true_col].std()
    y_pred_z = (df[pred_col] - df[pred_col].mean()) / df[pred_col].std()
    df['residual'] = y_true_z - y_pred_z
    
    # 3. 拆解 Sentiment：Panic vs. Confidence
    # 假设：Sentiment Volume 高 = 恐慌 (Panic)
    # 假设：Sentiment Volume 低 = 平稳 (Calm/Confidence)
    median_sent = df['sentiment_index'].median()
    
    # 构造两个变量
    # Panic Force: 只有当情绪高于中位数时才有值，否则为0
    df['Panic_Force'] = df['sentiment_index'].apply(lambda x: x - median_sent if x > median_sent else 0)
    
    # Calm Force: 只有当情绪低于中位数时才有值 (取绝对值距离)
    df['Calm_Force']  = df['sentiment_index'].apply(lambda x: abs(x - median_sent) if x < median_sent else 0)
    
    # 4. 回归分析 (机制的核心)
    # 问：Panic 会推高 Residual 吗？ Calm 会拉低 Residual 吗？
    reg = LinearRegression()
    X = df[['Panic_Force', 'Calm_Force']]
    y = df['residual']
    reg.fit(X, y)
    
    coef_panic = reg.coef_[0]
    coef_calm  = reg.coef_[1]
    r_squared  = reg.score(X, y)
    
    print(f"\n======== 🔬 Mechanism Probe: {country_code} ========")
    print(f"Panic Coefficient: {coef_panic:.4f}")
    print(f"Calm  Coefficient: {coef_calm:.4f}")
    print(f"R-squared (Explanatory Power): {r_squared:.4f}")
    
    return coef_panic, coef_calm, r_squared

# ================= 主程序 =================
if __name__ == "__main__":
    p_usa, t_usa, p_chn, t_chn, sent = load_data()
    
    # 处理中国列名
    if 'pCN_weighted' not in p_chn.columns:
         p_chn['pCN_weighted'] = p_chn['pS'] * 0.596 + p_chn['pN'] * 0.404
    col_usa = 'yhat' if 'yhat' in p_usa.columns else p_usa.columns[1]
    
    # 运行机制分析
    res_usa = analyze_mechanism(p_usa, t_usa, sent, 'USA', 'num_inc', col_usa)
    res_chn = analyze_mechanism(p_chn, t_chn, sent, 'CHN', 'national_ili_weighted', 'pCN_weighted')
    
    # === 画出最终的 Figure 5 ===
    labels = ['Panic Impact\n(>Median)', 'Confidence Impact\n(<Median)']
    usa_vals = [res_usa[0], res_usa[1]]
    chn_vals = [res_chn[0], res_chn[1]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(9, 6))
    plt.bar(x - width/2, usa_vals, width, label='USA (Buffered)', color='lightgray', edgecolor='gray')
    plt.bar(x + width/2, chn_vals, width, label='China (Coupled)', color=['#ff6666', '#66cc66'], edgecolor='black')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.ylabel('Impact on Healthcare Burden (Residuals)')
    plt.title('Mechanism Verification: Asymmetric Regulation vs. Buffering', fontweight='bold')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.savefig('./result/Figure5_Mechanism.png', dpi=300)
    print("\n✅ Figure 5 生成完成。这才是我们要在论文里讲的故事！")