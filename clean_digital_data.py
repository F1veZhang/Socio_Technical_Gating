import pandas as pd
import numpy as np

def clean_digital_data():
    print("正在开始数据清洗...")

    # ================= 1. World Bank Data (全球互联网普及率) =================
    # 跳过前4行元数据
    try:
        wb_df = pd.read_csv('./data/Individuals_using_the_Internet.csv', skiprows=4)
        
        # 筛选中美两国
        wb_clean = wb_df[wb_df['Country Name'].isin(['China', 'United States'])].copy()
        
        # 宽表转长表 (Melt)
        wb_clean = wb_clean.melt(id_vars=['Country Name'], 
                                 value_vars=[str(y) for y in range(1960, 2024)], # 根据文件实际年份调整
                                 var_name='Year', value_name='WB_Internet_Penetration')
        
        wb_clean['Year'] = pd.to_numeric(wb_clean['Year'], errors='coerce')
        wb_clean = wb_clean.dropna(subset=['Year'])
        wb_clean['Year'] = wb_clean['Year'].astype(int)
        
        # 重新透视：一行一年，列为国家
        wb_clean = wb_clean.pivot(index='Year', columns='Country Name', values='WB_Internet_Penetration').reset_index()
        wb_clean.columns = ['Year', 'WB_China_Internet', 'WB_USA_Internet']
        print("✅ World Bank 数据处理完成")
    except Exception as e:
        print(f"⚠️ World Bank 数据处理出错: {e}")
        wb_clean = pd.DataFrame(columns=['Year', 'WB_China_Internet', 'WB_USA_Internet'])

    # ================= 2. Pew Research (美国互联网使用) =================
    try:
        usa_internet = pd.read_csv('./data/USA_internet_use_data_2025-11-20.csv', skiprows=3) # 根据文件实际跳过行数
        
        # 清洗年份 (处理脚注导致的非数字行)
        usa_internet['Year'] = pd.to_numeric(usa_internet['Year'], errors='coerce')
        usa_internet = usa_internet.dropna(subset=['Year'])
        usa_internet['Year'] = usa_internet['Year'].astype(int)
        
        # 清洗百分比 (去除 %)
        usa_internet['Pew_USA_Internet'] = usa_internet['U.S. adults'].astype(str).str.replace('%', '').astype(float)
        
        # 按年份聚合 (可能有同一年多次调查)
        usa_internet_clean = usa_internet.groupby('Year')['Pew_USA_Internet'].mean().reset_index()
        print("✅ Pew (Internet) 数据处理完成")
    except Exception as e:
        print(f"⚠️ Pew (Internet) 数据处理出错: {e}")
        usa_internet_clean = pd.DataFrame(columns=['Year', 'Pew_USA_Internet'])

    # ================= 3. Pew Research (美国手机拥有率) =================
    try:
        usa_mobile = pd.read_csv('./data/USA_mobile_phone_ownership_data_2025-11-20.csv', skiprows=3)
        
        # 清洗日期转年份
        usa_mobile['Date_Obj'] = pd.to_datetime(usa_mobile['Year'], errors='coerce')
        usa_mobile = usa_mobile.dropna(subset=['Date_Obj'])
        usa_mobile['Year'] = usa_mobile['Date_Obj'].dt.year
        
        # 清洗百分比列 (Cellphone & Smartphone)
        for col in ['Cellphone', 'Smartphone']:
            # 处理空值和特殊字符
            usa_mobile[col] = usa_mobile[col].astype(str).str.strip()
            usa_mobile[col] = pd.to_numeric(usa_mobile[col].str.replace('%', ''), errors='coerce')
            
        usa_mobile_clean = usa_mobile.groupby('Year')[['Cellphone', 'Smartphone']].mean().reset_index()
        usa_mobile_clean.columns = ['Year', 'Pew_USA_Cellphone', 'Pew_USA_Smartphone']
        print("✅ Pew (Mobile) 数据处理完成")
    except Exception as e:
        print(f"⚠️ Pew (Mobile) 数据处理出错: {e}")
        usa_mobile_clean = pd.DataFrame(columns=['Year', 'Pew_USA_Cellphone', 'Pew_USA_Smartphone'])

    # ================= 4. CNNIC (中国互联网络信息中心) =================
    try:
        # 假设文件名是 csv 格式
        china_internet = pd.read_excel('./data/China_internet_use.xlsx')
        
        # 清洗日期
        china_internet['date'] = pd.to_datetime(china_internet['time'], errors='coerce')
        china_internet = china_internet.dropna(subset=['date'])
        china_internet['Year'] = china_internet['date'].dt.year
        
        # 转换普及率 (rate 是 0.x 格式，转为百分比)
        china_internet['CNNIC_China_Internet'] = china_internet['rate'] * 100
        
        china_internet_clean = china_internet.groupby('Year')['CNNIC_China_Internet'].mean().reset_index()
        print("✅ CNNIC 数据处理完成")
    except Exception as e:
        print(f"⚠️ CNNIC 数据处理出错: {e}")
        china_internet_clean = pd.DataFrame(columns=['Year', 'CNNIC_China_Internet'])

    # ================= 5. 合并数据 (Merge) =================
    # 使用 Outer Join 保证不丢失任何一年的数据
    merged_df = wb_clean.merge(usa_internet_clean, on='Year', how='outer') \
                        .merge(usa_mobile_clean, on='Year', how='outer') \
                        .merge(china_internet_clean, on='Year', how='outer')
    
    # 排序并筛选 2000 年之后
    merged_df = merged_df.sort_values('Year')
    merged_df = merged_df[merged_df['Year'] >= 2000]
    
    # 填充：如果某一年 WB 数据缺失，可以用 Pew/CNNIC 补齐（可选，这里先保留原始空值）
    
    # 保存结果
    output_file = './data/cleaned_digital_infrastructure.csv'
    merged_df.to_csv(output_file, index=False)
    
    print("\n" + "="*40)
    print(f"🎉 清洗完成！文件已保存为: {output_file}")
    print("="*40)
    print("数据预览 (2015-2025):")
    print(merged_df[merged_df['Year'] >= 2015].head(15))

if __name__ == "__main__":
    clean_digital_data()