import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np

# ================= 1. 数据加载与预处理 =================

# 读取数据
df_chn = pd.read_csv('./data/aligned_data_china_complete.csv')
df_usa = pd.read_csv('./data/aligned_data_usa_complete.csv')

# 转换时间格式
df_chn['date'] = pd.to_datetime(df_chn['date'])
df_usa['date'] = pd.to_datetime(df_usa['date'])

# --- 核心步骤：计算平滑趋势 (Smoothing) ---
# 使用 6 周滑动平均 (Rolling Mean) 来提取长期趋势
window_size = 6 

# 中国
df_chn['sentiment_smooth'] = df_chn['sentiment_index'].rolling(window=window_size, center=True).mean()
# 美国
df_usa['sentiment_smooth'] = df_usa['sentiment_index'].rolling(window=window_size, center=True).mean()

# ================= 2. 绘图代码 (High-Quality Visualization) =================

plt.style.use('seaborn-v0_8-white')
# 创建两个子图，共享X轴
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# 颜色定义
color_sent_raw = '#ff9999'   # 浅红色：原始散点/细线
color_sent_trend = '#D62728' # 深红色：平滑趋势线
color_epi = '#1F77B4'        # 蓝色：疫情区域

# -----------------------------------------------------------
# 子图 A: 美国 (USA)
# -----------------------------------------------------------
ax1_twin = ax1.twinx()

# 1. 绘制疫情 (右轴, 蓝色面积图)
# 填充区域
ax1_twin.fill_between(df_usa['date'], 0, df_usa['num_inc'], 
                      color=color_epi, alpha=0.25, label='Reported Flu Cases')
# 蓝色轮廓线
ax1_twin.plot(df_usa['date'], df_usa['num_inc'], 
              color=color_epi, linewidth=1, alpha=0.6)
ax1_twin.set_ylabel('Weekly Reported Cases (USA)', color=color_epi, fontsize=12, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor=color_epi)

# 2. 绘制舆情 (左轴, 红色线)
# (A) 原始波动 (浅色细线)
ax1.plot(df_usa['date'], df_usa['sentiment_index'], 
         color=color_sent_raw, linewidth=0.8, alpha=0.5, label='Raw Sentiment (Weekly)')
# (B) 平滑趋势 (深色粗线)
ax1.plot(df_usa['date'], df_usa['sentiment_smooth'], 
         color=color_sent_trend, linewidth=2.5, label=f'{window_size}-Week Smoothed Trend')

ax1.set_ylabel('Vaccine Sentiment Index (Twitter)', color=color_sent_trend, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_sent_trend)

# 标题与装饰
ax1.set_title('A. United States: Ideological Sentiment vs. Epidemic Reality (2015-2025)', 
              loc='left', fontsize=16, fontweight='bold', pad=15)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5) # 0值基准线

# -----------------------------------------------------------
# 子图 B: 中国 (China)
# -----------------------------------------------------------
ax2_twin = ax2.twinx()

# 1. 绘制疫情 (右轴, 蓝色面积图) - 使用人口加权后的 ILI%
ax2_twin.fill_between(df_chn['date'], 0, df_chn['national_ili_weighted'], 
                      color=color_epi, alpha=0.25, label='National ILI% (Weighted)')
ax2_twin.plot(df_chn['date'], df_chn['national_ili_weighted'], 
              color=color_epi, linewidth=1, alpha=0.6)
ax2_twin.set_ylabel('National ILI % (China)', color=color_epi, fontsize=12, fontweight='bold')
ax2_twin.tick_params(axis='y', labelcolor=color_epi)

# 2. 绘制舆情 (左轴, 红色线)
# (A) 原始波动
ax2.plot(df_chn['date'], df_chn['sentiment_index'], 
         color=color_sent_raw, linewidth=0.8, alpha=0.5, label='Raw Sentiment (Weekly)')
# (B) 平滑趋势
ax2.plot(df_chn['date'], df_chn['sentiment_smooth'], 
         color=color_sent_trend, linewidth=2.5, label=f'{window_size}-Week Smoothed Trend')

ax2.set_ylabel('Sentiment Index (Weibo)', color=color_sent_trend, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_sent_trend)

# 标题与装饰
ax2.set_title('B. China: Logistic Sentiment vs. Epidemic Reality (2015-2025)', 
              loc='left', fontsize=16, fontweight='bold', pad=15)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# -----------------------------------------------------------
# 公共装饰 (Highlights & Legends)
# -----------------------------------------------------------

# 1. 标注 COVID-19 时期 (灰色背景)
for ax in [ax1, ax2]:
    ax.axvspan(pd.to_datetime('2020-01-01'), pd.to_datetime('2023-01-01'), 
               color='gray', alpha=0.1, zorder=0, label='COVID-19 Pandemic')

# 2. X轴格式
ax2.set_xlabel('Year', fontsize=14)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.set_xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2025-09-01'))

# 3. 图例 (Legend)
# 合并左右轴的图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_1t, labels_1t = ax1_twin.get_legend_handles_labels()
ax1.legend(lines_1 + lines_1t, labels_1 + labels_1t, loc='upper left', frameon=True, fancybox=True, framealpha=0.9)

lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_2t, labels_2t = ax2_twin.get_legend_handles_labels()
ax2.legend(lines_2 + lines_2t, labels_2 + labels_2t, loc='upper left', frameon=True, fancybox=True, framealpha=0.9)

plt.tight_layout()

# 保存高清大图
plt.savefig('Figure1_Pulse_of_Two_Nations_Smoothed.png', dpi=300, bbox_inches='tight')
print("图表已生成: Figure1_Pulse_of_Two_Nations_Smoothed.png")
plt.show()