import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date
import pandas as pd # تأكد من استيراد pandas

# إعدادات الصفحة
st.set_page_config(page_title="توقعات أسعار الفضة", layout="wide")
st.title('🔮 تطبيق توقع أسعار الفضة باستخدام Prophet')

# المدخلات
n_years = st.slider('عدد سنوات البيانات للتدريب:', 1, 10, 5)
period = n_years * 365
days_to_predict = st.slider('عدد الأيام للتوقع:', 30, 365, 90)

# --- التعديل هنا (الحل للمشكلة) ---
@st.cache_data
def load_data(ticker):
    start_date = (date.today().replace(year=date.today().year - n_years)).strftime('%Y-%m-%d')
    end_date = date.today().strftime('%Y-%m-%d')
    
    # تحميل البيانات
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # ⚠️ الخطوة المهمة: إصلاح مشكلة MultiIndex في yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('جاري تحميل البيانات...')
data = load_data("SI=F")
data_load_state.text('تم التحميل! ✅')

st.subheader('البيانات الخام')
st.write(data.tail())

# --- تجهيز البيانات لـ Prophet ---
# نتأكد أننا نأخذ العمود الصحيح
df_train = data[['Date', 'Close']].copy() # نستخدم copy لتجنب التحذيرات
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# التأكد من إزالة التوقيت الزمني (Timezone)
df_train['ds'] = pd.to_datetime(df_train['ds']).dt.tz_localize(None)

# تدريب النموذج
st.subheader('جاري التدريب...')
m = Prophet()
m.fit(df_train)

# التوقع
future = m.make_future_dataframe(periods=days_to_predict)
forecast = m.predict(future)

# العرض
st.subheader(f'توقعات الـ {days_to_predict} يوم القادمة')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("تحليل الاتجاه")
fig2 = m.plot_components(forecast)
st.write(fig2)
