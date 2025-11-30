import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(page_title="Exti AI Admission", page_icon="🎓", layout="wide")

# استایل CSS
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif; direction: rtl; text-align: right; }
    .uni-card {
        background-color: #1E1E1E; padding: 15px; border-radius: 10px;
        border: 1px solid #333; margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .uni-card:hover { transform: scale(1.02); border-color: #00ADB5; }
    .stButton button { background-color: #00ADB5; color: white; font-size: 18px; width: 100%; }
    .alert-box { background-color: #ff4b4b; color: white; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# لود منابع
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('real_admission_model.pkl')
        db = pd.read_csv('universities_db_ready.csv')
        return model, db
    except:
        return None, None

model, df_unis = load_resources()

if model is None:
    st.error("❌ فایل مدل یا دیتابیس یافت نشد.")
    st.stop()

# تابع فیلتر تنوع
def get_diverse_selection(df, sort_by='Rank'):
    if df.empty: return df
    if sort_by == 'Rank':
        df_sorted = df.sort_values(by=['Rank'], ascending=True)
    else:
        df_sorted = df.sort_values(by=['Chance', 'Rank'], ascending=[False, True])
        
    selected_indices = []
    unique_countries = df_sorted['Country'].unique()
    for country in unique_countries:
        country_unis = df_sorted[df_sorted['Country'] == country]
        top_2 = country_unis.head(2)
        selected_indices.extend(top_2.index.tolist())
    
    final_df = df.loc[selected_indices]
    
    if sort_by == 'Rank':
        return final_df.sort_values(by=['Rank'], ascending=True)
    else:
        return final_df.sort_values(by=['Chance'], ascending=False)

# رابط کاربری
st.title("🎓 دستیار هوشمند اپلای Exti")
st.markdown("تحلیل دقیق شانس پذیرش با **فیلترهای منطقی و واقعی**")

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    with c1: gpa = st.number_input("معدل کل (0-20)", 10.0, 20.0, 17.86)
    with c2: ielts = st.number_input("نمره آیلتس", 0.0, 9.0, 6.5, step=0.5)
    with c3: papers = st.number_input("تعداد مقاله", 0, 50, 1)
    with c4: 
        degree_label = st.selectbox("مقطع هدف", ["لیسانس (Bachelor)", "ارشد (Master)", "دکتری (PhD)"])
        level_map = {"لیسانس (Bachelor)": 0, "ارشد (Master)": 1, "دکتری (PhD)": 2}
        level_code = level_map[degree_label]
    
    c5, c6 = st.columns(2)
    with c5:
        prev_uni_tier = st.selectbox("سطح دانشگاه قبلی", 
                                     ["سطح ۱ (شریف/تهران/امیرکبیر...)", "سطح ۲ (دولتی مراکز استان)", "سطح ۳ (سایر)"])
        prev_uni_map = {"سطح ۱ (شریف/تهران/امیرکبیر...)": 1, "سطح ۲ (دولتی مراکز استان)": 2, "سطح ۳ (سایر)": 3}
        prev_uni_code = prev_uni_map[prev_uni_tier]
    with c6:
        st.write("")
        st.write("")
        gre_check = st.checkbox("مدرک GRE دارم")
        gre_code = 1 if gre_check else 0

# دکمه جستجو
if st.button("🔮 جستجوی هوشمند"):
    
    hard_reject = False
    reject_reason = ""

    # قانون ۱: نمره زبان منطقی
    min_ielts_required = 6.5 if level_code == 2 else 6.0 if level_code == 1 else 5.0
    if ielts < min_ielts_required:
        hard_reject = True
        reject_reason = f"❌ نمره آیلتس شما ({ielts}) کمتر از حد نصاب اولیه ({min_ielts_required}) برای این مقطع است."

    # قانون ۲: معدل منطقی
    min_gpa_required = 15.0 if level_code == 2 else 14.0 if level_code == 1 else 12.0
    if gpa < min_gpa_required:
        hard_reject = True
        reject_reason = f"❌ معدل شما ({gpa}) کمتر از حداقل مورد نیاز ({min_gpa_required}) برای این مقطع است."

    if hard_reject:
        st.error(reject_reason)
        st.info("💡 پیشنهاد مشاور: می‌توانید برای دوره‌های کالج زبان یا مقاطع پایین‌تر اقدام کنید.")
    
    else:
        with st.spinner("در حال تحلیل و اعمال فیلترهای سخت‌گیرانه..."):
            time.sleep(0.5)
            
            candidates = df_unis.copy()
            
            user_data = pd.DataFrame({
                'GPA': [gpa] * len(candidates),
                'IELTS': [ielts] * len(candidates),
                'Papers': [papers] * len(candidates),
                'Level': [level_code] * len(candidates),
                'Prev_Uni': [prev_uni_code] * len(candidates),
                'GRE': [gre_code] * len(candidates),
                'Difficulty': candidates['Difficulty']
            })
            
            try:
                probs = model.predict_proba(user_data)[:, 1] * 100
                candidates['Chance'] = probs
            except:
                st.error("خطا در مدل.")
                st.stop()
            
            # جریمه‌های داینامیک منطقی
            if ielts < 6.5:
                mask_top20 = candidates['Rank'] <= 20
                candidates.loc[mask_top20, 'Chance'] -= 30

                mask_top50 = (candidates['Rank'] > 20) & (candidates['Rank'] <= 50)
                candidates.loc[mask_top50, 'Chance'] -= 20

            if level_code == 2:
                if papers == 0:
                    candidates['Chance'] -= 30
                elif papers == 1:
                    mask_top50 = candidates['Rank'] <= 50
                    candidates.loc[mask_top50, 'Chance'] -= 10

            candidates['Chance'] = candidates['Chance'].clip(lower=0, upper=99).round(1)

            # حداقل شانس برای نمایش: ۵٪
            candidates = candidates[candidates['Chance'] >= 5]

            # دسته‌بندی
            raw_dreams = candidates[(candidates['Chance'] >= 5) & (candidates['Chance'] < 40)]
            raw_targets = candidates[(candidates['Chance'] >= 40) & (candidates['Chance'] < 75)]
            raw_safeties = candidates[candidates['Chance'] >= 75]
            
            final_dreams = get_diverse_selection(raw_dreams, sort_by='Chance')
            final_targets = get_diverse_selection(raw_targets, sort_by='Rank')
            final_safeties = get_diverse_selection(raw_safeties, sort_by='Rank')

            st.markdown("---")
            t1, t2, t3 = st.tabs([
                f"🎯 انتخاب‌های منطقی ({len(final_targets)})", 
                f"🌟 رویاپردازانه ({len(final_dreams)})", 
                f"🛡️ سوپاپ اطمینان ({len(final_safeties)})"
            ])
            
            def show(df, color):
                if df.empty: 
                    st.info("موردی در این دسته یافت نشد.")
                else:
                    for _, row in df.head(30).iterrows():
                        tuition = row.get('Tuition_Type', '')
                        res_score = int(row['Research_Score']) if 'Research_Score' in row else 'N/A'
                        
                        st.markdown(f"""
                        <div class="uni-card" style="border-right: 5px solid {color};">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <h4 style="margin:0; color:white;">{row['University']}</h4>
                                <span style="background:{color}; color:#000; padding:2px 8px; border-radius:10px; font-size:12px; font-weight:bold;">
                                    {row['Chance']}%
                                </span>
                            </div>
                            <div style="color:#aaa; font-size:13px; margin-top:5px;">
                                📍 {row['Country']} | 🏆 رنک: {int(row['Rank'])} | 📚 امتیاز پژوهشی: {res_score}
                                <br>
                                <span style="color:#888; font-size:11px;">{tuition}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            with t1: show(final_targets, "#00ADB5")
            with t2: show(final_dreams, "#FFA500")
            with t3: show(final_safeties, "#00FF7F")
