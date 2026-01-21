import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./data/cleaned_digital_infrastructure.csv')

plt.style.use('seaborn-v0_8-white')
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制中国互联网普及率 
ax.plot(df['Year'], df['CNNIC_China_Internet'], 
        color='#D62728', linewidth=3, marker='o', label='China: Internet Penetration (CNNIC)')

# 绘制美国智能手机普及率 
ax.plot(df['Year'], df['Pew_USA_Smartphone'], 
        color='#1f77b4', linewidth=3, linestyle='--', marker='s', label='USA: Smartphone Ownership (Pew)')

# 绘制 65% 阈值线
ax.axhline(65, color='gray', linestyle=':', linewidth=2, label='Critical Mass Threshold (65%)')

# 标注关键区域
# 1. 中国跨越点
cross_point = df[df['CNNIC_China_Internet'] >= 65].iloc[0]
ax.annotate(f"China Crosses 65%\n(Start of High Coupling)\nYear: {int(cross_point['Year'])}", 
            xy=(cross_point['Year'], 65), xytext=(cross_point['Year']-2, 80),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11, fontweight='bold', color='#D62728')

# 2. 美国状态
ax.text(2015, 75, "USA: Always Above Threshold\n(Coupling is Event-Driven)", 
        fontsize=11, fontweight='bold', color='#1f77b4')

# 装饰
ax.set_title('The Material Basis of Sentiment-Epidemic Coupling (2010-2025)', fontsize=14, fontweight='bold')
ax.set_ylabel('Penetration Rate (%)', fontsize=12)
ax.set_xlim(2010, 2025)
ax.set_ylim(30, 100)
ax.legend(loc='lower right', frameon=True)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()