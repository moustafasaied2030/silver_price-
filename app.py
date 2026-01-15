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
st.set_page_config(page_title="توقعات الفضة (EGP)", layout="wide")

st.title('🔮 منصة توقع أسعار الفضة (بالجنيه المصري)')
st.markdown("""
يقوم هذا النظام بدمج سعر الفضة العالمي (USD) مع سعر صرف الدولار (EGP) 
لإعطاء توقع دقيق للسعر المحلي في مصر.
""")
st.markdown("---")

# ---------------------------------------------------------
# 2. القائمة الجانبية
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ إعدادات التوقع")
    
    # اختيار تاريخ معين
    default_date = date.today() + timedelta(days=30)
    target_date = st.date_input(
        "📅 اختر التاريخ المستهدف للتوقع:",
        min_value=date.today() + timedelta(days=1),
        value=default_date
    )
    
    st.markdown("---")
    st.write("🔧 إعدادات النموذج:")
    n_years = st.slider('عدد سنوات البيانات التاريخية:', 1, 10, 5)

# ---------------------------------------------------------
# 3. دالة المعالجة المتقدمة (دمج الفضة مع الدولار)
# ---------------------------------------------------------
@st.cache_data
def load_and_predict_egp(years, target_d):
    # تحديد الفترة
    start_date = (date.today().replace(year=date.today().year - years)).strftime('%Y-%m-%d')
    end_date = date.today().strftime('%Y-%m-%d')
    
    # --- الخطوة 1: جلب سعر الفضة العالمي ---
    silver_data = yf.download("SI=F", start=start_date, end=end_date, progress=False)
    silver_data.reset_index(inplace=True)
    
    # تنظيف بيانات الفضة
    if isinstance(silver_data.columns, pd.MultiIndex):
        silver_data.columns = silver_data.columns.get_level_values(0)
    
    # --- الخطوة 2: جلب سعر الدولار مقابل الجنيه ---
    egp_data = yf.download("EGP=X", start=start_date, end=end_date, progress=False)
    egp_data.reset_index(inplace=True)
    
    # تنظيف بيانات الدولار
    if isinstance(egp_data.columns, pd.MultiIndex):
        egp_data.columns = egp_data.columns.get_level_values(0)

    # --- الخطوة 3: الدمج والحساب ---
    # نجهز جدولين بسيطين للدمج
    df_silver = silver_data[['Date', 'Close']].rename(columns={'Close': 'Silver_USD'})
    df_egp = egp_data[['Date', 'Close']].rename(columns={'Close': 'USD_EGP'})
    
    # التأكد من تنسيق التاريخ
    df_silver['Date'] = pd.to_datetime(df_silver['Date']).dt.tz_localize(None)
    df_egp['Date'] = pd.to_datetime(df_egp['Date']).dt.tz_localize(None)

    # دمج الجدولين بناء على التاريخ
    merged_df = pd.merge(df_silver, df_egp, on='Date', how='inner')
    
    # حساب السعر بالجنيه (سعر الفضة * سعر الدولار)
    # ملاحظة: سعر الفضة عالمياً للأونصة
    merged_df['Price_EGP'] = merged_df['Silver_USD'] * merged_df['USD_EGP']
    
    # تجهيز البيانات لـ Prophet
    df_train = pd.DataFrame()
    df_train['ds'] = merged_df['Date']
    df_train['y'] = merged_df['Price_EGP']
    
    # حذف القيم المفقودة
    df_train.dropna(inplace=True)

    # --- الخطوة 4: التدريب ---
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    
    # --- الخطوة 5: التوقع ---
    days_diff = (target_d - date.today()).days
    future_days = days_diff + 10
    
    future = m.make_future_dataframe(periods=future_days)
    forecast = m.predict(future)
    
    return m, forecast

# ---------------------------------------------------------
# 4. واجهة العرض
# ---------------------------------------------------------
try:
    with st.spinner('جاري تحليل سعر الفضة وسعر الدولار ودمج البيانات...'):
        model, forecast = load_and_predict_egp(n_years, target_date)

    # --- القسم الأول: السعر المتوقع ---
    st.subheader(f"💰 سعر أونصة الفضة المتوقع (EGP) يوم: {target_date}")
    
    prediction_row = forecast[forecast['ds'].dt.date == target_date]
    
    if not prediction_row.empty:
        price = prediction_row['yhat'].values[0]
        lower = prediction_row['yhat_lower'].values[0]
        upper = prediction_row['yhat_upper'].values[0]
        
        st.markdown(f"""
        <div style="
            background-color: #fff3e0; 
            padding: 20px; 
            border-radius: 15px; 
            border: 2px solid #ffb74d; 
            text-align: center; 
            margin-bottom: 25px;">
            <h1 style="color: #e65100; margin:0; font-size: 50px;">{price:,.2f} ج.م</h1>
            <p style="color: #666; margin-top: 5px;">
                للأونصة (Ounce)
            </p>
            <p style="font-size: 14px; color: #888;">
                (النطاق المتوقع: {lower:,.2f} - {upper:,.2f})
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # تحويل تقريبي للجرام (عيار 999)
        # الأونصة = 31.1035 جرام
        gram_price = price / 31.1035
        st.info(f"💡 هذا يعني أن سعر الجرام (عيار 999) المتوقع تقريباً: **{gram_price:,.2f} ج.م**")
        
    else:
        st.warning(f"لم يتم العثور على توقع لهذا التاريخ.")

    # --- القسم الثاني: الرسم البياني ---
    st.subheader("📈 مسار السعر بالجنيه المصري")
    fig_main = plot_plotly(model, forecast)
    fig_main.update_layout(yaxis_title="السعر (EGP)", xaxis_title="التاريخ")
    st.plotly_chart(fig_main, use_container_width=True)

    st.markdown("---")

    # --- القسم الثالث: المكونات ---
    st.subheader("📊 تحليل الاتجاهات (Trend)")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

except Exception as e:
    st.error("حدث خطأ أثناء المعالجة:")
    st.code(str(e))
