import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date, timedelta
import pandas as pd

# إعدادات الصفحة
st.set_page_config(page_title="توقعات أسعار الفضة", layout="wide")
st.title('🔮 تطبيق توقع أسعار الفضة باستخدام Prophet')

# ---------------------------------------------------------
# الجزء الجديد: اختيار التواريخ
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    # تاريخ بداية البيانات (الافتراضي قبل 5 سنوات من اليوم)
    default_start = date.today() - timedelta(days=5*365)
    start_date_input = st.date_input(
        "اختر تاريخ بداية البيانات التاريخية للتدريب:",
        value=default_start,
        max_value=date.today()
    )

with col2:
    # تاريخ نهاية التوقع (الافتراضي بعد 90 يوم من اليوم)
    default_end = date.today() + timedelta(days=90)
    forecast_end_date = st.date_input(
        "اختر التاريخ الذي تريد التوقع حتى وصوله:",
        value=default_end,
        min_value=date.today() + timedelta(days=1)
    )

# حساب عدد الأيام للتوقع بناءً على التاريخ المختار
days_to_predict = (forecast_end_date - date.today()).days

# ---------------------------------------------------------
# دالة التحميل المعدلة
# ---------------------------------------------------------
@st.cache_data
def load_data(ticker, start_d):
    end_d = date.today().strftime('%Y-%m-%d')
    s_d = start_d.strftime('%Y-%m-%d')
    
    # تحميل البيانات
    data = yf.download(ticker, start=s_d, end=end_d)
    
    # إصلاح مشكلة الأعمدة (MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.reset_index(inplace=True)
    return data

# تحميل البيانات بناءً على التاريخ المختار
data_load_state = st.text('جاري تحميل البيانات...')
try:
    data = load_data("SI=F", start_date_input)
    
    if data.empty:
        st.error("لم يتم العثور على بيانات في هذا التاريخ. يرجى اختيار تاريخ أقدم.")
        st.stop()
        
    data_load_state.text('تم التحميل! ✅')

    # عرض البيانات الخام
    with st.expander("عرض البيانات الخام"):
        st.write(data.tail())

    # ---------------------------------------------------------
    # تجهيز النموذج
    # ---------------------------------------------------------
    df_train = data[['Date', 'Close']].copy()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    df_train['ds'] = pd.to_datetime(df_train['ds']).dt.tz_localize(None)

    # التدريب
    m = Prophet()
    m.fit(df_train)

    # التوقع
    future = m.make_future_dataframe(periods=days_to_predict)
    forecast = m.predict(future)

    # ---------------------------------------------------------
    # العرض
    # ---------------------------------------------------------
    st.subheader(f'توقعات الأسعار حتى {forecast_end_date}')
    
    # الرسم البياني
    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(
        title="توقع سعر الفضة (USD)",
        yaxis_title="السعر",
        xaxis_title="التاريخ"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # عرض السعر المتوقع في اليوم المحدد
    predicted_val = forecast.loc[forecast['ds'].dt.date == forecast_end_date]['yhat'].values
    if len(predicted_val) > 0:
        st.metric(label=f"السعر المتوقع يوم {forecast_end_date}", value=f"${predicted_val[0]:.2f}")

    st.subheader("تحليل المكونات")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

except Exception as e:
    st.error(f"حدث خطأ: {e}")
