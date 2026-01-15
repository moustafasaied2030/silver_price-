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
st.markdown("---")

# ---------------------------------------------------------
# 2. القائمة الجانبية
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ إعدادات التوقع")
    
    # اختيار تاريخ معين للمعرفة السعر عنده
    # نجعل الافتراضي بعد شهر من الآن
    default_date = date.today() + timedelta(days=30)
    
    target_date = st.date_input(
        "📅 اختر التاريخ المستهدف للتوقع:",
        min_value=date.today() + timedelta(days=1),
        value=default_date
    )
    
    st.markdown("---")
    st.write("🔧 إعدادات النموذج:")
    n_years = st.slider('عدد سنوات البيانات التاريخية للتدريب:', 1, 15, 5)

# ---------------------------------------------------------
# 3. دالة المعالجة والذكاء الاصطناعي (تم إصلاح المشاكل هنا)
# ---------------------------------------------------------
@st.cache_data
def load_and_predict(years, target_d):
    # 1. تحديد فترة البيانات
    start_date = (date.today().replace(year=date.today().year - years)).strftime('%Y-%m-%d')
    end_date = date.today().strftime('%Y-%m-%d')
    
    # 2. تحميل البيانات
    data = yf.download("SI=F", start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)

    # 3. تنظيف البيانات (الحل الجذري لمشكلة yfinance)
    # نقوم بإنشاء DataFrame جديد ونظيف تماماً
    df = pd.DataFrame()
    
    # التأكد من عمود التاريخ
    if 'Date' in data.columns:
        df['ds'] = data['Date']
    else:
        # أحيانا يكون التاريخ هو الـ index بعد التحديثات
        df['ds'] = data.index

    # التأكد من عمود السعر (Close)
    # نحاول الوصول لعمود Close بأي طريقة كانت سواء كان MultiIndex أو عادي
    try:
        # محاولة الوصول المباشر
        close_data = data['Close']
    except KeyError:
        # محاولة الوصول عن طريق xs في حالة MultiIndex المعقد
        try:
            close_data = data.xs('Close', axis=1, level=0)
        except:
            # إذا فشل كل شيء، نأخذ العمود الثاني (عادة هو السعر بعد التاريخ)
            close_data = data.iloc[:, 1]

    # تحويل البيانات إلى أرقام وإجبارها أن تكون 1D array (حل مشكلة TypeError)
    if isinstance(close_data, pd.DataFrame):
        df['y'] = close_data.iloc[:, 0].values
    else:
        df['y'] = close_data.values

    # إزالة المنطقة الزمنية
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    
    # حذف أي صفوف فارغة قد تسبب مشاكل
    df.dropna(inplace=True)

    # 4. تدريب النموذج
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    
    # 5. التوقع
    # حساب الفرق بالأيام
    days_diff = (target_d - date.today()).days
    
    # هامش أمان: نضيف 10 أيام زيادة للتأكد من أن التاريخ المستهدف موجود داخل التوقع
    # (هذا يحل مشكلة "التاريخ بعيد جداً")
    future_days = days_diff + 10
    
    future = m.make_future_dataframe(periods=future_days)
    forecast = m.predict(future)
    
    return m, forecast

# ---------------------------------------------------------
# 4. واجهة العرض
# ---------------------------------------------------------
try:
    with st.spinner('جاري جلب البيانات وتدريب النموذج...'):
        model, forecast = load_and_predict(n_years, target_date)

    # --- القسم الأول: عرض السعر ---
    st.subheader(f"💰 السعر المتوقع ليوم: {target_date}")
    
    # البحث عن التاريخ في النتائج
    # نستخدم dt.date للمقارنة الصحيحة
    prediction_row = forecast[forecast['ds'].dt.date == target_date]
    
    if not prediction_row.empty:
        price = prediction_row['yhat'].values[0]
        lower = prediction_row['yhat_lower'].values[0]
        upper = prediction_row['yhat_upper'].values[0]
        
        # تصميم كارت السعر
        st.markdown(f"""
        <div style="
            background-color: #e8f5e9; 
            padding: 20px; 
            border-radius: 15px; 
            border: 1px solid #c8e6c9; 
            text-align: center; 
            margin-bottom: 25px;">
            <h1 style="color: #2e7d32; margin:0; font-size: 50px;">${price:.2f}</h1>
            <p style="color: #666; margin-top: 5px;">
                النطاق المتوقع: بين <b>{lower:.2f}</b> و <b>{upper:.2f}</b> دولار
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # رسالة خطأ أوضح في حال عدم وجود التاريخ
        st.warning(f"لم يتم العثور على توقع لهذا التاريخ ({target_date}). حاول اختيار تاريخ أقرب قليلاً.")

    # --- القسم الثاني: الرسم البياني ---
    st.subheader("📈 مسار السعر (الماضي + المستقبل)")
    fig_main = plot_plotly(model, forecast)
    fig_main.update_layout(yaxis_title="سعر الفضة (USD)", xaxis_title="التاريخ")
    st.plotly_chart(fig_main, use_container_width=True)

    st.markdown("---")

    # --- القسم الثالث: التحليل الفني ---
    st.subheader("📊 العوامل المؤثرة (Trend & Seasonality)")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

except Exception as e:
    # عرض الأخطاء بشكل مفصل للمساعدة في الحل
    st.error("حدث خطأ غير متوقع:")
    st.code(str(e))
