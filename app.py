import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Прогноз дохода",
    page_icon="💼",
    layout="centered"
)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ----------------------------
# DICTIONARIES (UI → MODEL)
# ----------------------------
workclass_map = {
    "Частная компания": "Private",
    "ИП (без наёмных работников)": "Self-emp-not-inc",
    "ИП (с наёмными работниками)": "Self-emp-inc",
    "Федеральное правительство": "Federal-gov",
    "Местные органы власти": "Local-gov",
    "Региональные органы власти": "State-gov",
    "Без оплаты": "Without-pay",
    "Никогда не работал(а)": "Never-worked"
}

education_map = {
    "Детский сад / дошкольное": "Preschool",
    "1–4 классы": "1st-4th",
    "5–6 классы": "5th-6th",
    "7–8 классы": "7th-8th",
    "9 класс": "9th",
    "10 класс": "10th",
    "11 класс": "11th",
    "Окончил(а) старшую школу": "HS-grad",
    "Некоторое высшее (без степени)": "Some-college",
    "Среднее специальное (прикладное)": "Assoc-voc",
    "Среднее специальное (академическое)": "Assoc-acdm",
    "Бакалавриат": "Bachelors",
    "Магистратура": "Masters",
    "Профессиональное образование": "Prof-school",
    "Докторантура": "Doctorate"
}

marital_map = {
    "Женат / замужем": "Married-civ-spouse",
    "Разведён(а)": "Divorced",
    "Никогда не был(а) в браке": "Never-married",
    "Раздельное проживание": "Separated",
    "Вдовец / вдова": "Widowed",
    "Супруг(а) отсутствует": "Married-spouse-absent"
}

occupation_map = {
    "Техническая поддержка": "Tech-support",
    "Ремесло / ремонт": "Craft-repair",
    "Сфера услуг": "Other-service",
    "Продажи": "Sales",
    "Руководитель / менеджер": "Exec-managerial",
    "Профессиональный специалист": "Prof-specialty",
    "Разнорабочий / уборка": "Handlers-cleaners",
    "Оператор станков": "Machine-op-inspct",
    "Административный персонал": "Adm-clerical",
    "Сельское хозяйство / рыболовство": "Farming-fishing",
    "Транспортировка": "Transport-moving",
    "Домашний персонал": "Priv-house-serv",
    "Охранные службы": "Protective-serv",
    "Вооружённые силы": "Armed-Forces"
}

relationship_map = {
    "Жена": "Wife",
    "Муж": "Husband",
    "Собственный ребёнок": "Own-child",
    "Не состоит в семье": "Not-in-family",
    "Другой родственник": "Other-relative",
    "Не женат / не замужем": "Unmarried"
}

race_map = {
    "Белый": "White",
    "Чёрный": "Black",
    "Азиатско-тихоокеанский регион": "Asian-Pac-Islander",
    "Коренной американец": "Amer-Indian-Eskimo",
    "Другое": "Other"
}

sex_map = {
    "Мужской": "Male",
    "Женский": "Female"
}

# ----------------------------
# UI
# ----------------------------
st.title("💼 Прогноз годового дохода")

st.write(
    """
    Приложение определяет, **превысит ли годовой доход человека $50 000**,  
    на основе демографических и профессиональных характеристик.
    """
)

st.markdown("---")

with st.form("input_form"):
    st.subheader("Введите данные")

    age = st.number_input("Возраст", 17, 100, 35)
    fnlwgt = st.number_input("Финальный вес (fnlwgt)", 0, value=200000)
    education_num = st.number_input("Уровень образования (числовой)", 1, 16, 10)
    capital_gain = st.number_input("Доход от капитала", 0, value=0)
    capital_loss = st.number_input("Убыток от капитала", 0, value=0)
    hours_per_week = st.number_input("Часов работы в неделю", 1, 100, 40)

    workclass_ru = st.selectbox("Тип занятости", list(workclass_map.keys()))
    education_ru = st.selectbox("Образование", list(education_map.keys()))
    marital_ru = st.selectbox("Семейное положение", list(marital_map.keys()))
    occupation_ru = st.selectbox("Профессия", list(occupation_map.keys()))
    relationship_ru = st.selectbox("Семейная роль", list(relationship_map.keys()))
    race_ru = st.selectbox("Раса", list(race_map.keys()))
    sex_ru = st.selectbox("Пол", list(sex_map.keys()))

    submitted = st.form_submit_button("🔍 Получить прогноз")

# ----------------------------
# PREDICTION
# ----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "fnlwgt": fnlwgt,
        "education-num": education_num,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "workclass": workclass_map[workclass_ru],
        "education": education_map[education_ru],
        "marital-status": marital_map[marital_ru],
        "occupation": occupation_map[occupation_ru],
        "relationship": relationship_map[relationship_ru],
        "race": race_map[race_ru],
        "sex": sex_map[sex_ru]
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.success(f"✅ **Доход превысит $50 000**\n\nВероятность: **{probability:.2%}**")
    else:
        st.error(f"❌ **Доход не превысит $50 000**\n\nВероятность: **{probability:.2%}**")
