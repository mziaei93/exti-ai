import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. لود کردن دیتابیس
try:
    df_unis = pd.read_csv('universities_db_ready.csv')
except:
    print("❌ دیتابیس یافت نشد.")
    exit()

print("🧠 در حال آموزش مدل اصلاح شده (نسخه منصفانه و بدون باگ)...")

data = []

# تولید ۵۰,۰۰۰ داده برای آموزش
for _ in range(50000):
    # --- پروفایل دانشجو ---
    level = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4]) # 0=Bachelor, 1=Master, 2=PhD
    
    # دانشگاه قبلی (1=Top Tier مثل تهران/شریف)
    prev_uni_tier = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3]) 
    
    gpa = np.clip(np.random.normal(16, 2.0), 12, 20)
    ielts = np.clip(np.random.normal(6.5, 1.0), 5.0, 9.0)
    
    # شانس داشتن مقاله
    papers = 0
    if level == 2: # PhD
        papers = np.random.choice([0, 1, 2, 3, 5], p=[0.4, 0.3, 0.2, 0.08, 0.02])
    elif level == 1: # Master
        papers = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])

    has_gre = np.random.choice([0, 1], p=[0.8, 0.2])

    # --- انتخاب دانشگاه هدف ---
    target = df_unis.sample(1).iloc[0]
    difficulty = target['Difficulty']
    
    # --- فرمول جدید (Tuned Logic) ---
    score = 0
    
    # 1. ضریب دانشگاه مبدأ (قدرت مدرک دانشگاه تهران)
    if prev_uni_tier == 1:
        gpa_boost = 1.25 
    elif prev_uni_tier == 2:
        gpa_boost = 1.0
    else:
        gpa_boost = 0.9
        
    adjusted_gpa = gpa * gpa_boost
    
    if level == 2: # PhD Logic
        paper_score = papers * 2.5
        gre_bonus = 2 if has_gre else 0
        
        score = (adjusted_gpa * 0.5) + (ielts * 0.8) + paper_score + gre_bonus
        
        # 🔴 اصلاح شد: نام متغیر یکسان شد
        threshold_mult = 2.1 
        
    elif level == 1: # Master Logic
        score = (adjusted_gpa * 0.9) + (ielts * 1.0) + (papers * 1.5)
        
        # 🔴 اصلاح شد
        threshold_mult = 1.9
        
    else: # Bachelor
        score = (adjusted_gpa * 1.2) + (ielts * 1.0)
        
        # 🔴 اصلاح شد
        threshold_mult = 1.8

    # شرط قبولی
    chance_noise = np.random.normal(0, 1.0)
    
    # محاسبه آستانه نهایی
    final_threshold = difficulty * threshold_mult
    
    # مقایسه امتیاز با آستانه
    admitted = 1 if (score + chance_noise) > final_threshold else 0

    data.append([gpa, ielts, papers, level, prev_uni_tier, has_gre, difficulty, admitted])

# 3. آموزش مدل
df_train = pd.DataFrame(data, columns=['GPA', 'IELTS', 'Papers', 'Level', 'Prev_Uni', 'GRE', 'Difficulty', 'Admitted'])

X = df_train[['GPA', 'IELTS', 'Papers', 'Level', 'Prev_Uni', 'GRE', 'Difficulty']]
y = df_train['Admitted']

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

joblib.dump(model, 'real_admission_model.pkl')
print("✅ مدل جدید با موفقیت ذخیره شد.")