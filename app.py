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
st.set_page_config(page_title="توقعات الفضة في مصر", layout="wide")

st.title('🔮 مؤشر الفضة المصري (صناعة يدوية)')
st.markdown("""
بما أن yfinance لا يدعم السعر المحلي، يقوم هذا التطبيق بحسابه كالتالي:
**السعر العالمي ($) × سعر الدولار (ج.م) ÷ 31.1 (للتحويل لجرام)**
""")
st.markdown("---")

# ---------------------------------------------------------
# 2. القائمة الجانبية
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    # اختيار تاريخ معين
    default_date = date.today() + timedelta(days=7)
    target_date = st.date_input(
        "📅 اختر يوم التوقع:",
        min_value=date.today() + timedelta(days=1),
        value=default_date
    )
    
    n_years = st.slider('سنوات البيانات السابقة:', 1, 10, 5)

# ---------------------------------------------------------
# 3. دالة بناء المؤشر المصري (الحل الذكي)
# ---------------------------------------------------------
@st.cache_data
def load_egyptian_silver_price(years, target_d):
    # تحديد الفترة
    start_date = (date.today().replace(year=date.today().year - years)).strftime('%Y-%m-%d')
    end_date = date.today().strftime('%Y-%m-%d')
    
    # 1. جلب الفضة العالمية
    silver = yf.download("SI=F", start=start_date, end=end_date, progress=False)
    if isinstance(silver.columns, pd.MultiIndex): silver.columns = silver.columns.get_level_values(0)
    silver.reset_index(inplace=True)
    
    # 2. جلب سعر الدولار
    usd_egp = yf.download("EGP=X", start=start_date, end=end_date, progress=False)
    if isinstance(usd_egp.columns, pd.MultiIndex): usd_egp.columns = usd_egp.columns.get_level_values(0)
    usd_egp.reset_index(inplace=True)

    # 3. دمج الجدولين (التاريخ مع التاريخ)
    # تنظيف الأسماء
    df_s = silver[['Date', 'Close']].rename(columns={'Close': 'Silver_USD'})
    df_u = usd_egp[['Date', 'Close']].rename(columns={'Close': 'USD_Rate'})
    
    # توحيد التوقيت
    df_s['Date'] = pd.to_datetime(df_s['Date']).dt.tz_localize(None)
    df_u['Date'] = pd.to_datetime(df_u['Date']).dt.tz_localize(None)
    
    # الدمج
    df = pd.merge(df_s, df_u, on='Date', how='inner')
    
    # 4. المعادلة السحرية (صناعة السعر المصري)
    # السعر = (سعر الأونصة بالدولار * سعر الدولار) / 31.1035 (للحصول على سعر الجرام)
    df['Price_EGP_Gram'] = (df['Silver_USD'] * df['USD_Rate']) / 31.1035
    
    # تجهيز لـ Prophet
    df_train = df[['Date', 'Price_EGP_Gram']].rename(columns={'Date': 'ds', 'Price_EGP_Gram': 'y'})
    df_train.dropna(inplace=True)
    
    # التدريب
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    
    # التوقع
    days_diff = (target_d - date.today()).days
    future = m.make_future_dataframe(periods=days_diff + 5)
    forecast = m.predict(future)
    
    return m, forecast

# ---------------------------------------------------------
# 4. العرض
# ---------------------------------------------------------
try:
    with st.spinner('جاري حساب سعر الجرام المصري بناءً على السعر العالمي والدولار...'):
        model, forecast = load_egyptian_silver_price(n_years, target_date)

    # جلب القيمة للتاريخ المختار
    target_row = forecast[forecast['ds'].dt.date == target_date]
    
    if not target_row.empty:
        price = target_row['yhat'].values[0]
        lower = target_row['yhat_lower'].values[0]
        upper = target_row['yhat_upper'].values[0]
        
        st.subheader(f"💎 سعر جرام الفضة (خام) المتوقع ليوم {target_date}")
        
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #0068c9;">
            <h1 style="color: #0068c9; font-size: 45px; margin:0;">{price:.2f} ج.م</h1>
            <p style="color: gray;">سعر الجرام (عيار 999 تقريباً)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("⚠️ تنبيه هام: هذا السعر هو سعر 'الخام' (Spot Price) ولا يشمل المصنعية أو مكسب التاجر.")
        
    else:
        st.error("التاريخ المختار بعيد جداً.")

    st.markdown("---")
    
    st.subheader("📈 رسم بياني لسعر جرام الفضة بالمصري")
    fig1 = plot_plotly(model, forecast)
    fig1.update_layout(yaxis_title="سعر الجرام (EGP)", xaxis_title="التاريخ")
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("📊 تحليل الاتجاهات")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

except Exception as e:
    st.error(f"حدث خطأ: {e}")
