import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date

# إعدادات الصفحة
st.set_page_config(page_title="توقعات أسعار الفضة", layout="wide")

st.title('🔮 تطبيق توقع أسعار الفضة باستخدام Prophet')
st.markdown("يستخدم هذا التطبيق الذكاء الاصطناعي لتوقع أسعار الفضة بناءً على البيانات التاريخية.")

# اختيار عدد سنوات البيانات للتدريب
n_years = st.slider('عدد سنوات البيانات التاريخية للتدريب:', 1, 10, 5)
period = n_years * 365

# اختيار عدد أيام التوقع للمستقبل
days_to_predict = st.slider('عدد الأيام المراد توقعها في المستقبل:', 30, 365, 90)

@st.cache_data
def load_data(ticker):
    # جلب البيانات من اليوم الحالي رجوعاً للوراء
    start_date = (date.today().replace(year=date.today().year - n_years)).strftime('%Y-%m-%d')
    end_date = date.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

# تحميل البيانات
data_load_state = st.text('جاري تحميل بيانات الفضة (SI=F)...')
data = load_data("SI=F")
data_load_state.text('تم تحميل البيانات بنجاح! ✅')

# عرض البيانات الخام
st.subheader('البيانات التاريخية الأخيرة')
st.write(data.tail())

# ---------------------------------------------------------
# التجهيز لـ Prophet
# ---------------------------------------------------------
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
# إزالة التوقيت الزمني
df_train['ds'] = df_train['ds'].dt.tz_localize(None)

# تدريب النموذج
st.subheader('جاري تدريب النموذج وعمل التوقعات...')
m = Prophet()
m.fit(df_train)

# عمل التوقعات
future = m.make_future_dataframe(periods=days_to_predict)
forecast = m.predict(future)

# ---------------------------------------------------------
# عرض النتائج
# ---------------------------------------------------------
st.subheader(f'توقعات الأسعار لـ {days_to_predict} يوم قادم')

# الرسم البياني التفاعلي باستخدام Plotly
fig1 = plot_plotly(m, forecast)
fig1.update_layout(title="توقع سعر الفضة (USD)", yaxis_title="السعر", xaxis_title="التاريخ")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("مكونات التوقع (الاتجاه والموسمية)")
fig2 = m.plot_components(forecast)
st.write(fig2)