import pandas as pd
import re
import langid
from tqdm import tqdm

# ================= 配置区域 =================
# 输入文件名
WEIBO_FILE = './data/Weibo.xlsx'      # 微博源文件
TWITTER_FILE = './data/Twitter.csv'   # 推特源文件

# 输出文件名 (三个国家分开存，方便后续做图)
OUT_CHINA = './data/dataset_china_weibo.csv'
OUT_USA = './data/dataset_usa_twitter.csv'
OUT_JAPAN = './data/dataset_japan_twitter.csv'

# ==========================================
# PROFESSIONAL KEYWORD CONFIGURATION (2015-2025)
# Sources: 
# 1. China CDC Guidelines & User's Supplementary Material
# 2. Broniatowski et al. (2013) for US English
# 3. Aramaki et al. (2011) for Japanese Context
# ==========================================

# 1. 纳入关键词 (Inclusion List)
# 逻辑：一条推文必须包含至少一个关键词
KEYWORDS_DICT = {
    'zh': [
        # Disease Names
        '流感', '流行性感冒', '甲流', '乙流', '禽流感', 'H1N1', 'H3N2', 'H5N1', 'H7N9', 'H9N2',
        # Vaccines & Interventions
        '疫苗', '流感针', '预防针', '接种',
        # Drugs (Covering 2015-2025 evolution)
        '奥司他韦', '达菲', '玛巴洛沙韦', '速福达', '连花清瘟', '抗病毒'
    ],
    'en': [
        # Disease Names
        'flu', 'influenza', 'h1n1', 'h3n2', 'h5n1', 'h5n6', 'h7n9', 'avian flu', 'bird flu', 'swine flu',
        # Vaccines (Including slang)
        'vaccine', 'vaccination', 'vax', 'anti-vax', 'flu shot', 'flu jab', 'immunization',
        # Drugs
        'tamiflu', 'oseltamivir', 'fluzone', 'flucelvax'
    ],
    'ja': [
        # Disease Names (Standard & Abbrev.)
        'インフル', 'インフルエンザ', '鳥インフル', 'H1N1', 'A型', 'B型',
        # Vaccines
        'ワクチン', '予防接種',
        # Drugs (Specific to Japan market)
        'タミフル', 'ゾフルーザ', 'イナビル', 'リレンザ' 
    ]
}

# 2. 排除关键词 (Exclusion/Stopwords List)
# 逻辑：如果推文包含任意一个词，直接剔除 (High Precision Filtering)
SPAM_KEYWORDS = [
    # --- Chinese (Weibo) ---
    '包邮', '代购', '下单', '淘宝', '天猫', '京东', '拼多多', '优惠券', '满减', # E-commerce
    '转发抽奖', '粉丝福利', '点击链接', '私信', '招代理', '刷单', # Spam/Bot
    '禽流感色', '流感妆', # Specific noise
    
    # --- English (Twitter) ---
    'giveaway', 'crypto', 'nft', 'bitcoin', 'ethereum', 'airdrop', # Crypto Spam
    'free shipping', 'discount', '% off', 'storewide', 'promo code', # Sales
    'check out my', 'subscribe', 'onlyfans', # Bot patterns
    'bieber fever', # Cultural noise
    
    # --- Japanese (Twitter) ---
    'プレゼント', 'キャンペーン', '当選', '応募', '拡散希望', # Giveaways
    '無料', '割引', 'セール', # Sales
    'インフルエンサー', # CRITICAL: "Influencer" (confuses with Influenza)
    'アマギフ' # (Amazon Gift Card spam)
]

# ================= 工具函数 =================

def clean_text_basic(text):
    """通用基础清洗"""
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)   # 去链接
    text = re.sub(r'@\w+', '', text)      # 去@用户
    text = re.sub(r'<.*?>', '', text)     # 去HTML
    text = text.replace('收起d', '')      # 去微博特有噪音
    return text.strip()

def is_spam(text):
    """检查是否含营销词"""
    text_lower = text.lower()
    for kw in SPAM_KEYWORDS:
        if kw in text_lower:
            return True
    return False

def check_relevance(text, lang):
    """检查是否包含对应语言的流感/疫苗关键词"""
    keywords = KEYWORDS_DICT.get(lang, [])
    text_lower = text.lower()
    for k in keywords:
        if k in text_lower:
            return True
    return False

# ================= 主处理逻辑 =================

def process_weibo():
    print(f"\n>>> 正在处理 [中国/微博] 数据...")
    try:
        df = pd.read_excel(WEIBO_FILE)
        # 假设列名是 '发布内容'
        df = df.dropna(subset=['发布内容'])
        
        # 1. 基础清洗
        df['clean_text'] = df['发布内容'].apply(clean_text_basic)
        
        # 2. 筛选中文 + 非广告 + 相关性
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = row['clean_text']
            if len(text) < 4: continue
            if is_spam(text): continue
            
            # 语言识别 (确保是中文)
            lang, _ = langid.classify(text)
            if lang != 'zh': continue
            
            # 关键词匹配
            if check_relevance(text, 'zh'):
                results.append(row)
                
        # 导出
        df_clean = pd.DataFrame(results)
        df_clean.to_csv(OUT_CHINA, index=False, encoding='utf-8-sig')
        print(f"   [中国] 清洗完成，剩余: {len(df_clean)} 条 -> {OUT_CHINA}")
        
    except Exception as e:
        print(f"   微博处理出错: {e}")

def process_twitter():
    print(f"\n>>> 正在处理 [美国&日本/推特] 数据...")
    try:
        # 读取CSV
        try:
            df = pd.read_csv(TWITTER_FILE, encoding='utf-8', on_bad_lines='skip')
        except:
            df = pd.read_csv(TWITTER_FILE, encoding='latin1', on_bad_lines='skip')
            
        # 假设内容列是 '发布内容', 语言列是 '语言类型'(如果有) 或者我们自己识别
        col_text = '发布内容'
        
        usa_rows = []
        japan_rows = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = clean_text_basic(str(row.get(col_text, '')))
            if len(text) < 4: continue
            if is_spam(text): continue
            
            # 语言识别 (核心步骤)
            # 如果csv自带 '语言类型' 列且准确，可以直接用，这里演示用langid重新识别更稳妥
            lang, conf = langid.classify(text)
            
            # 分流逻辑
            if lang == 'en':
                # 检查英文关键词 -> 归入美国库
                if check_relevance(text, 'en'):
                    row['clean_text'] = text
                    usa_rows.append(row)
            
            elif lang == 'ja':
                # 检查日文关键词 -> 归入日本库
                if check_relevance(text, 'ja'):
                    row['clean_text'] = text
                    japan_rows.append(row)
        
        # 导出美国数据
        pd.DataFrame(usa_rows).to_csv(OUT_USA, index=False)
        print(f"   [美国] 清洗完成，剩余: {len(usa_rows)} 条 -> {OUT_USA}")
        
        # 导出日本数据
        pd.DataFrame(japan_rows).to_csv(OUT_JAPAN, index=False, encoding='utf-8-sig')
        print(f"   [日本] 清洗完成，剩余: {len(japan_rows)} 条 -> {OUT_JAPAN}")

    except Exception as e:
        print(f"   推特处理出错: {e}")

# ================= 执行 =================
if __name__ == "__main__":
    # 先确保安装了 langid: pip install langid
    process_weibo()
    process_twitter()