import pandas as pd
import re
import datetime
import os

# ================= 配置区域 =================
# 文件路径
FILE_SENTIMENT = './data/weekly_sentiment_series_FINAL.csv'
FILE_CHINA_HIST = './data/ILI_percent_clean.csv'
FILE_CHINA_NEW = './data/南北方省份最新.xlsx'
FILE_USA_FLU = './data/USA_ILIPredict.csv' 

# 输出路径
OUT_CHINA = './data/aligned_data_china_complete.csv'
OUT_USA = './data/aligned_data_usa_complete.csv'

# ================= 科学权重 =================
WEIGHT_SOUTH = 0.596
WEIGHT_NORTH = 0.404

# ================= 智能读取工具 =================

def parse_chinese_week(s):
    match = re.search(r'(\d{4})年第(\d+)周', str(s))
    if match:
        year = int(match.group(1))
        week = int(match.group(2))
        return pd.to_datetime(f"{year}-W{week}-1", format='%Y-W%W-%w', errors='coerce')
    return pd.NaT

def read_file_smart(file_path):
    if not os.path.exists(file_path):
        print(f"   ❌ 错误: 找不到文件 {file_path}")
        return None
    
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext in ['.xlsx', '.xls']:
        try:
            print(f"   检测到 Excel 格式，正在读取: {os.path.basename(file_path)}")
            return pd.read_excel(file_path)
        except Exception as e:
            print(f"   ❌ Excel 读取失败: {e}")
            return None
    else:
        encodings = ['utf-8', 'gb18030', 'gbk', 'utf-8-sig', 'latin1']
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                if df.shape[1] > 1:
                    print(f"   成功读取 CSV {os.path.basename(file_path)} (编码: {enc})")
                    return df
            except:
                continue
        print(f"   ❌ 无法识别 CSV 编码: {file_path}")
        return None

def align_datasets():
    print(">>> 开始最终数据对齐 (修复列名版)...")
    
    # 1. 加载情感数据
    df_sent = read_file_smart(FILE_SENTIMENT)
    if df_sent is None: return
    df_sent['date'] = pd.to_datetime(df_sent['date'])
    
    # 2. 处理中国数据
    print("   处理中国数据...")
    df_chn_hist = read_file_smart(FILE_CHINA_HIST)
    df_chn_hist['date_str'] = df_chn_hist['year'].astype(str) + '-W' + df_chn_hist['week'].astype(str) + '-1'
    df_chn_hist['date'] = pd.to_datetime(df_chn_hist['date_str'], format='%Y-W%W-%w', errors='coerce')
    
    df_chn_new = read_file_smart(FILE_CHINA_NEW)
    df_chn_new['date'] = df_chn_new['日期'].apply(parse_chinese_week)
    df_chn_new = df_chn_new.rename(columns={'南方省份ILI%（%）': 'south_prov', '北方省份ILI%（%）': 'north_prov'})
    
    cols = ['date', 'south_prov', 'north_prov']
    df_chn_full = pd.concat([df_chn_hist[cols], df_chn_new[cols]], ignore_index=True)
    df_chn_full = df_chn_full.dropna(subset=['date']).sort_values('date')
    
    df_chn_full['south_prov'] = pd.to_numeric(df_chn_full['south_prov'], errors='coerce')
    df_chn_full['north_prov'] = pd.to_numeric(df_chn_full['north_prov'], errors='coerce')
    df_chn_full['national_ili_weighted'] = (df_chn_full['south_prov'] * WEIGHT_SOUTH + 
                                            df_chn_full['north_prov'] * WEIGHT_NORTH)
    
    # 3. 处理美国数据
    print("   处理美国数据...")
    df_usa_flu = read_file_smart(FILE_USA_FLU)
    if df_usa_flu is None: return
    
    # 自动识别 country 列
    country_cols = [c for c in df_usa_flu.columns if c.lower() == 'country']
    if country_cols:
        print(f"   筛选国家: USA")
        df_usa_flu = df_usa_flu[df_usa_flu[country_cols[0]] == 'USA'].copy()
    else:
        print("   ⚠️ 未发现 'country' 列，默认全量为美国数据")
    
    # --- 核心修复：自动识别病例数据列名 ---
    # 你的文件列名是 num_ILI_patient
    possible_cols = ['num_ILI_patient', 'num_inc', 'predicted_mean', 'inc', 'cases', 'ILI']
    inc_col = None
    
    for col in possible_cols:
        # 不区分大小写匹配
        matches = [c for c in df_usa_flu.columns if c.lower() == col.lower()]
        if matches:
            inc_col = matches[0]
            break
            
    if inc_col:
        print(f"   ✅ 成功识别病例数据列: {inc_col}")
    else:
        print(f"   ❌ 错误: 无法在 {df_usa_flu.columns.tolist()} 中找到病例数据列！")
        return

    # 统一重命名为 num_inc 方便后续处理
    df_usa_flu = df_usa_flu.rename(columns={inc_col: 'num_inc'})
    df_usa_flu['num_inc'] = pd.to_numeric(df_usa_flu['num_inc'], errors='coerce').fillna(0)
    
    # 识别时间列 (YEAR, WEEK)
    # 处理列名大小写问题 (比如 year vs YEAR)
    year_col = [c for c in df_usa_flu.columns if c.lower() == 'year'][0]
    week_col = [c for c in df_usa_flu.columns if c.lower() == 'week'][0]
    
    df_usa_agg = df_usa_flu.groupby([year_col, week_col])['num_inc'].sum().reset_index()
    
    df_usa_agg['date_str'] = df_usa_agg[year_col].astype(str) + '-W' + df_usa_agg[week_col].astype(str) + '-1'
    df_usa_agg['date'] = pd.to_datetime(df_usa_agg['date_str'], format='%Y-W%W-%w', errors='coerce')
    df_usa_agg = df_usa_agg.dropna(subset=['date']).sort_values('date')

    # 4. 对齐合并
    sent_cn = df_sent[df_sent['country'] == 'CHN'].sort_values('date')
    merged_chn = pd.merge_asof(sent_cn, df_chn_full, on='date', direction='nearest', tolerance=pd.Timedelta(days=6))
    
    sent_us = df_sent[df_sent['country'] == 'USA'].sort_values('date')
    merged_usa = pd.merge_asof(sent_us, df_usa_agg, on='date', direction='nearest', tolerance=pd.Timedelta(days=6))
    
    # 5. 导出
    merged_chn.to_csv(OUT_CHINA, index=False, encoding='utf-8-sig') 
    merged_usa.to_csv(OUT_USA, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 对齐完成！")
    print(f"   中国数据: {OUT_CHINA}")
    print(f"   美国数据: {OUT_USA}")

if __name__ == "__main__":
    align_datasets()