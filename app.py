import streamlit as st
import yfinance as yf
from prophet import Prophet
from datetime import date, timedelta
import pandas as pd

# إعدادات الصفحة لتكون بسيطة ومركزة
st.set_page_config(page_title="حاسبة سعر الفضة", layout="centered")

st.title('💰 حاسبة سعر الفضة المتوقع')
st.markdown("---")

# 1. إدخال التاريخ المستهدف فقط
target_date = st.date_input(
    "📅 اختر اليوم الذي تريد معرفة السعر المتوقع فيه:",
    min_value=date.today() + timedelta(days=1),
    value=date.today() + timedelta(days=1)
)

# دالة التحميل والتدريب (تعمل في الخلفية)
@st.cache_data
def predict_price(target_d):
    # تحميل بيانات آخر 3 سنوات (كافية للسرعة والدقة)
    start_date = (date.today() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    end_date = date.today().strftime('%Y-%m-%d')
    
    data = yf.download("SI=F", start=start_date, end=end_date, progress=False)
    
    # إصلاح مشكلة الأعمدة
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.reset_index(inplace=True)
    
    # تجهيز وتدريب
    df = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    
    m = Prophet()
    m.fit(df)
    
    # التوقع حتى التاريخ المطلوب
    days_diff = (target_d - date.today()).days
    if days_diff < 1: days_diff = 1
    
    future = m.make_future_dataframe(periods=days_diff)
    forecast = m.predict(future)
    
    # استخراج قيمة اليوم المحدد
    result = forecast[forecast['ds'].dt.date == target_d]
    return result

# 2. زر التنفيذ والعرض
if st.button('🔮 احسب السعر الآن', use_container_width=True):
    with st.spinner('جاري تحليل السوق وحساب السعر...'):
        try:
            prediction = predict_price(target_date)
            
            if not prediction.empty:
                price = prediction['yhat'].values[0]
                
                # --- عرض النتيجة فقط ---
                st.markdown("### السعر المتوقع هو:")
                st.markdown(f"""
                <div style="text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                    <h1 style="color: #0068c9; font-size: 60px; margin: 0;">${price:.2f}</h1>
                    <p style="color: grey; margin: 0;">دولار أمريكي للأونصة</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("لم يتمكن النموذج من حساب التاريخ المحدد.")
                
        except Exception as e:
            st.error(f"حدث خطأ: {e}")
