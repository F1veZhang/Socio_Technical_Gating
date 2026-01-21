import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import jieba
import re
from collections import Counter
import matplotlib.font_manager as fm
import os
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局配置
# ==========================================
FILE_ALIGNED_CHINA = './data/aligned_data_china_complete.csv'
FILE_ALIGNED_USA = './data/aligned_data_usa_complete.csv'
FILE_RAW_WEIBO = './data/Weibo.xlsx'    
FILE_RAW_TWITTER = './data/Twitter.csv' 

# 疫情分界线
CUTOFF_DATE = pd.Timestamp('2020-01-20')

# --- 字体自动配置 ---
def get_chinese_font_path():
    """尝试在系统中寻找可用的中文字体路径"""
    potential_paths = [
        r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc', 
        '/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/STHeiti Light.ttc'
    ]
    for path in potential_paths:
        if os.path.exists(path): return path
    for f in fm.fontManager.ttflist:
        if 'SimHei' in f.name or 'Microsoft YaHei' in f.name: return f.fname
    return None 

CHINESE_FONT_PATH = get_chinese_font_path()
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.unicode_minus'] = False 
if CHINESE_FONT_PATH:
    try:
        font_prop = fm.FontProperties(fname=CHINESE_FONT_PATH)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'Arial Unicode MS', 'SimHei']
    except: pass

# ==========================================
# 1. 数值分析模块
# ==========================================
def analyze_numerical_proofs():
    print("\n" + "="*50)
    print("STEP 1: 数值分析 (基于 aligned_data 计算斜率与趋势)")
    print("="*50)
    
    try:
        df_china = pd.read_csv(FILE_ALIGNED_CHINA)
        df_usa = pd.read_csv(FILE_ALIGNED_USA)
    except FileNotFoundError:
        print(f"[错误] 找不到文件: {FILE_ALIGNED_CHINA} 或 {FILE_ALIGNED_USA}")
        return

    df_china['date'] = pd.to_datetime(df_china['date'])
    df_usa['date'] = pd.to_datetime(df_usa['date'])
    df_china['Period'] = df_china['date'].apply(lambda x: 'Post-COVID' if x >= CUTOFF_DATE else 'Pre-COVID')
    df_usa['Period'] = df_usa['date'].apply(lambda x: 'Post-COVID' if x >= CUTOFF_DATE else 'Pre-COVID')

    def calc_metrics(df, x_col, y_col, vol_col):
        pre = df[df['Period']=='Pre-COVID'].dropna(subset=[x_col, y_col])
        post = df[df['Period']=='Post-COVID'].dropna(subset=[x_col, y_col])
        slope_pre = np.polyfit(pre[x_col], pre[y_col], 1)[0] if len(pre)>1 else 0
        slope_post = np.polyfit(post[x_col], post[y_col], 1)[0] if len(post)>1 else 0
        vol_pre = df[df['Period']=='Pre-COVID'][vol_col].mean()
        vol_post = df[df['Period']=='Post-COVID'][vol_col].mean()
        return slope_pre, slope_post, vol_pre, vol_post

    s_chn_pre, s_chn_post, v_chn_pre, v_chn_post = calc_metrics(df_china, 'sentiment_index', 'national_ili_weighted', 'post_volume')
    s_usa_pre, s_usa_post, v_usa_pre, v_usa_post = calc_metrics(df_usa, 'sentiment_index', 'num_inc', 'post_volume')

    print(f"\n[中国 China] 结果:")
    print(f"  > 转化斜率 (Slope): {s_chn_pre:.4f} (Pre) -> {s_chn_post:.4f} (Post)")
    print(f"  > 讨论量基准 (Volume): {v_chn_pre:.0f} -> {v_chn_post:.0f}")

    print(f"\n[美国 USA] 结果:")
    print(f"  > 转化斜率 (Slope): {s_usa_pre:.0f} (Pre) -> {s_usa_post:.0f} (Post)")
    print(f"  > 讨论量基准 (Volume): {v_usa_pre:.0f} -> {v_usa_post:.0f}")

    # 绘图
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
    sns.regplot(data=df_china[df_china['Period']=='Pre-COVID'], x='sentiment_index', y='national_ili_weighted', ax=axes1[0], color='gray', scatter_kws={'alpha':0.3}, label='Pre-COVID')
    sns.regplot(data=df_china[df_china['Period']=='Post-COVID'], x='sentiment_index', y='national_ili_weighted', ax=axes1[0], color='#D62728', scatter_kws={'alpha':0.3}, label='Post-COVID')
    axes1[0].set_title('China: Sensitization (Slope Change)', fontsize=14)
    axes1[0].legend()
    sns.regplot(data=df_usa[df_usa['Period']=='Pre-COVID'], x='sentiment_index', y='num_inc', ax=axes1[1], color='gray', scatter_kws={'alpha':0.3}, label='Pre-COVID')
    sns.regplot(data=df_usa[df_usa['Period']=='Post-COVID'], x='sentiment_index', y='num_inc', ax=axes1[1], color='#1f77b4', scatter_kws={'alpha':0.3}, label='Post-COVID')
    axes1[1].set_title('USA: Buffering (Stable Slope)', fontsize=14)
    axes1[1].legend()
    plt.tight_layout()
    plt.savefig('./result/result_1_conversion_rate.png', dpi=300)

    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
    axes2[0].plot(df_china['date'], df_china['post_volume'], color='#D62728', alpha=0.8)
    axes2[0].axvline(CUTOFF_DATE, color='black', linestyle='--')
    axes2[0].set_title('China: Volume Drift', fontsize=14)
    axes2[1].plot(df_usa['date'], df_usa['post_volume'], color='#1f77b4', alpha=0.8)
    axes2[1].axvline(CUTOFF_DATE, color='black', linestyle='--')
    axes2[1].set_title('USA: Volume Drift', fontsize=14)
    plt.tight_layout()
    plt.savefig('./result/result_2_baseline_drift.png', dpi=300)
    print("  > 数值分析图表已保存。")

# ==========================================
# 2. 语义分析模块
# ==========================================
def read_file_smart(filepath):
    print(f"  > 正在读取: {filepath} ...")
    if not os.path.exists(filepath):
        print(f"    [错误] 文件不存在！")
        return None
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        try: return pd.read_excel(filepath)
        except Exception as e: print(f"    [Excel读取失败] {e}"); return None
    encodings = ['utf-8', 'gb18030', 'gbk']
    for enc in encodings:
        try: return pd.read_csv(filepath, encoding=enc, low_memory=False)
        except: continue
    return None

def clean_text_chinese(text):
    text = str(text)
    if text == 'nan': return ''
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\S+#", "", text)
    text = re.sub(r"@[^ ]+", "", text)
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text) # 只保留中文
    return text

def clean_text_english(text):
    text = str(text)
    if text == 'nan': return ''
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^ ]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

# --- 关键修复：中文日期解析函数 ---
def parse_chinese_date(date_val):
    """处理 '2015年01月06日 08:20' 这种格式"""
    s = str(date_val)
    # 将年、月、日替换为标准分隔符
    s = s.replace('年', '-').replace('月', '-').replace('日', '')
    return pd.to_datetime(s, errors='coerce')

def get_top_keywords(text_list, is_cn=True, top_n=20):
    full_text = " ".join(text_list)
    if is_cn:
        words = jieba.lcut(full_text)
        stopwords = {'收起', '展开', '全文', '的', '了', '在', '是', '我', '有', '和', '就', '人', '都', '一', '个', '上', '也', '很', '到', '说', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '吗', '吧', '视频', '链接', '微博', '转发', '回复', '月', '日', '年', '时', '分', '秒', '他', '她', '它'}
        words = [w for w in words if len(w)>1 and w not in stopwords]
    else:
        words = full_text.split()
        stopwords = {'the', 'to', 'and', 'of', 'in', 'is', 'for', 'on', 'that', 'it', 'with', 'as', 'are', 'at', 'be', 'this', 'have', 'from', 'or', 'you', 'not', 'by', 'but', 'https', 'co', 'http', 'rt'}
        words = [w for w in words if len(w)>2 and w not in stopwords]
    return Counter(words), Counter(words).most_common(top_n)

def analyze_semantic_proofs():
    print("\n" + "="*50)
    print("STEP 2: 语义分析 (修复日期解析问题)")
    print("="*50)
    
    df_wb = read_file_smart(FILE_RAW_WEIBO)
    df_tw = read_file_smart(FILE_RAW_TWITTER)
    
    if df_wb is None or df_tw is None: return

    # 自动列名匹配
    def find_col(df, candidates):
        for col in df.columns:
            if col in candidates: return col
        return None

    col_time_wb = find_col(df_wb, ['发布时间', 'created_at', 'date', 'Time', 'time'])
    col_text_wb = find_col(df_wb, ['发布内容', 'text', 'content', 'Text', '微博内容', '博文内容'])
    col_time_tw = find_col(df_tw, ['发布时间', 'created_at', 'date', 'Time'])
    col_text_tw = find_col(df_tw, ['发布内容', 'text', 'content', 'Text'])

    if not col_time_wb or not col_text_wb: print(f"[错误] 微博缺少必要列: {df_wb.columns}"); return
    if not col_time_tw or not col_text_tw: print(f"[错误] 推特缺少必要列: {df_tw.columns}"); return

    # --- 修复点：应用自定义日期解析 ---
    print(f"  > 处理微博日期 (列: {col_time_wb})...")
    df_wb['date'] = df_wb[col_time_wb].apply(parse_chinese_date)
    # 打印几条日期看看是否成功
    print(f"    日期样例(转换后): {df_wb['date'].dropna().head(3).astype(str).tolist()}")

    print(f"  > 处理推特日期 (列: {col_time_tw})...")
    df_tw['date'] = pd.to_datetime(df_tw[col_time_tw], errors='coerce')

    # 处理逻辑
    def process_data(df, text_col, date_col, is_cn, label):
        print(f"\n>>> 分析 {label} ...")
        mask_pre = df[date_col] < CUTOFF_DATE
        mask_post = df[date_col] >= CUTOFF_DATE
        
        # 提取并清洗
        texts_pre = df[mask_pre][text_col].astype(str).apply(clean_text_chinese if is_cn else clean_text_english)
        texts_post = df[mask_post][text_col].astype(str).apply(clean_text_chinese if is_cn else clean_text_english)
        
        texts_pre = [t for t in texts_pre if len(t.strip()) > 0]
        texts_post = [t for t in texts_post if len(t.strip()) > 0]
        
        print(f"  有效样本数: Pre={len(texts_pre)} | Post={len(texts_post)}")
        
        freq_pre, top_pre = get_top_keywords(texts_pre, is_cn)
        freq_post, top_post = get_top_keywords(texts_post, is_cn)
        
        print(f"  Top 5 (Pre): {top_pre[:5]}")
        print(f"  Top 5 (Post): {top_post[:5]}")
        return freq_pre, freq_post

    freq_chn_pre, freq_chn_post = process_data(df_wb, col_text_wb, 'date', True, '中国微博')
    freq_usa_pre, freq_usa_post = process_data(df_tw, col_text_tw, 'date', False, '美国推特')

    # 绘图
    print("\n> 正在绘制词云...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    def plot_cloud(ax, freq, title, cmap, font=None):
        if not freq:
            ax.text(0.5, 0.5, "NO DATA", ha='center'); ax.set_title(title); ax.axis('off'); return
        wc = WordCloud(font_path=font, width=600, height=400, background_color='white', colormap=cmap, max_words=100)
        wc.generate_from_frequencies(freq)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    plot_cloud(axes[0,0], freq_chn_pre, 'China Weibo (Pre-COVID)', 'Blues', CHINESE_FONT_PATH)
    plot_cloud(axes[0,1], freq_chn_post, 'China Weibo (Post-COVID)', 'Reds', CHINESE_FONT_PATH)
    plot_cloud(axes[1,0], freq_usa_pre, 'USA Twitter (Pre-COVID)', 'Blues', None)
    plot_cloud(axes[1,1], freq_usa_post, 'USA Twitter (Post-COVID)', 'Reds', None)

    plt.tight_layout()
    plt.savefig('./result/result_3_semantic_wordclouds.png', dpi=300)
    print("  > 图表已保存: result_3_semantic_wordclouds.png")

if __name__ == "__main__":
    analyze_numerical_proofs()
    analyze_semantic_proofs()
    print("\n[完成] 所有步骤结束。")