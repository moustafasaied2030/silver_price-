import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. إعدادات الصفحة
# ---------------------------------------------------------
st.set_page_config(page_title="تحليل وتوقع الفضة الشامل", layout="wide")

st.title('🔮 منصة تحليل وتوقع أسعار الفضة (SI=F)')
st.markdown("""
هذا التطبيق يجمع بين:
1. **حاسبة الأسعار:** لتوقع السعر في يوم محدد.
2. **التحليل الفني:** لعرض الاتجاه العام (Trend) والموسمية (Seasonality).
""")
st.markdown("---")

# ---------------------------------------------------------
# 2. القائمة الجانبية (للمدخلات والتحكم)
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ إعدادات التوقع")
    
    # اختيار تاريخ معين للمعرفة السعر عنده
    target_date = st.date_input(
        "📅 اختر التاريخ المستهدف للتوقع:",
        min_value=date.today() + timedelta(days=1),
        value=date.today() + timedelta(days=30)
    )
    
    st.markdown("---")
    st.write("🔧 إعدادات النموذج:")
    n_years = st.slider('عدد سنوات البيانات التاريخية للتدريب:', 1, 15, 5)

# ---------------------------------------------------------
# 3. دالة جلب وتجهيز البيانات (مع إصلاح الأخطاء)
# ---------------------------------------------------------
@st.cache_data
def load_and_predict(years, target_d):
    # تحديد تواريخ التحميل
    start_date = (date.today().replace(year=date.today().year - years)).strftime('%Y-%m-%d')
    end_date = date.today().strftime('%Y-%m-%d')
    
    # تحميل البيانات
    data = yf.download("SI=F", start=start_date, end=end_date, progress=False)
    
    # إصلاح مشكلة MultiIndex في yfinance الجديد
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.reset_index(inplace=True)
    
    # تجهيز البيانات لـ Prophet
    df = data[['Date', 'Close']].copy()
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    
    # بناء النموذج وتدريبه
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    
    # حساب الفترة الزمنية حتى التاريخ المستهدف
    days_diff = (target_d - date.today()).days
    # إضافة هامش بسيط للتأكد من تغطية التاريخ
    if days_diff < 1: days_diff = 1
    
    future = m.make_future_dataframe(periods=days_diff)
    forecast = m.predict(future)
    
    return m, forecast

# ---------------------------------------------------------
# 4. التنفيذ والعرض
# ---------------------------------------------------------
try:
    with st.spinner('جاري جلب البيانات، تدريب الذكاء الاصطناعي، ورسم المخططات...'):
        model, forecast = load_and_predict(n_years, target_date)

    # --- أولاً: عرض السعر المتوقع (طلبك الخاص بالحاسبة) ---
    st.subheader(f"💰 السعر المتوقع ليوم: {target_date}")
    
    # استخراج السعر
    prediction_row = forecast[forecast['ds'].dt.date == target_date]
    
    if not prediction_row.empty:
        price = prediction_row['yhat'].values[0]
        lower = prediction_row['yhat_lower'].values[0]
        upper = prediction_row['yhat_upper'].values[0]
        
        # عرض جذاب للرقم
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #f9f9f9;">
                <h2 style="color: #333; margin:0;">${price:.2f}</h2>
                <p style="color: gray; font-size: 14px;">(المدى المتوقع: {lower:.2f} - {upper:.2f})</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("التاريخ المختار بعيد جداً أو غير صالح.")

    st.markdown("---")

    # --- ثانياً: الرسم البياني التفاعلي (Main Chart) ---
    st.subheader("📈 مسار السعر (البيانات التاريخية + التوقع)")
    fig_main = plot_plotly(model, forecast)
    fig_main.update_layout(yaxis_title="سعر الفضة (USD)", xaxis_title="التاريخ")
    st.plotly_chart(fig_main, use_container_width=True)

    st.markdown("---")

    # --- ثالثاً: رسومات Trend و Seasonality ---
    st.subheader("📊 تحليل المكونات (Trend & Seasonality)")
    st.info("هذه الرسومات توضح الاتجاه العام للسوق، وتأثير أيام الأسبوع، وتأثير شهور السنة على السعر.")
    
    # استخدام matplotlib لرسم المكونات
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

except Exception as e:
    st.error(f"حدث خطأ أثناء التشغيل: {e}")
    st.write("تفاصيل الخطأ:", e)
