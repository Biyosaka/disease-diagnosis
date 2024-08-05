import csv
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer

with open('steamlit_web/Logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('ssteamlit_web/scaler3 (1).pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('steamlit_web/symptom4.pkl', 'rb') as f:
    symptom_list = pickle.load(f)    

with open('steamlit_web/Gencoder.pk', 'rb') as f:
    encoder = pickle.load(f)    


symptom_name = pd.read_csv('Symptom_names.csv')

def preprocess_inputs(gender, age, weight, blood_pressure, temperature, symptoms):
    input_data = pd.DataFrame({
        'Gender': [1 if gender == 'Male' else 0],
        'Age': [age],
        'Weight': [weight],
        'Blood Pressure': [blood_pressure],
        'Temperature': [temperature],
        'symptoms': [symptoms]
    })

    input_data['Gender'] = encoder.fit_transform(input_data['Gender'])
    input_data['symptoms'] = input_data['symptoms'].apply(lambda x: x.replace(' ','').split(','))
    mlb = MultiLabelBinarizer(classes=symptom_name.columns)
    symptoms_encoded = pd.DataFrame(mlb.fit_transform(input_data['symptoms']), columns=mlb.classes_, index=input_data.index)
    
    symptoms_input = pd.concat([pd.DataFrame(columns=symptom_list), symptoms_encoded], axis=0)

    numerical_inputs = scaler.transform(input_data[['Gender','Age', 'Weight', 'Blood Pressure', 'Temperature']])
    numerical_inputs_df = pd.DataFrame(numerical_inputs, columns=['Gender', 'Age', 'Weight', 'Blood Pressure', 'Temperature'])
    
    processed_inputs = pd.concat([numerical_inputs_df.reset_index(drop=True), symptoms_input.reset_index(drop=True)], axis=1)
    return processed_inputs

def get_predict():
    processed_inputs = preprocess_inputs(gender, age, weight, blood_pressure, temperature, symptoms)
    # prediction = model.predict(processed_inputs)
    prediction = model.predict_proba(processed_inputs)

    top_3_indices = np.argsort(prediction[0])[-3:][::-1]
    top_3_probabilities = prediction[0][top_3_indices]
    top_3_classes = model.classes_[top_3_indices]
    return top_3_classes, top_3_probabilities

def get_info(n):
    global info_list
    disease_info = pd.read_csv('disease_info.csv')

    disease_info_dict = disease_info.set_index('disease').T.to_dict()
    predicted_disease = top_3_classes[n]
    disease_details = disease_info_dict.get(predicted_disease, {})
    return disease_details
   
def is_symptom(symptoms, symptom_list):
    inputs = [s.strip() for s in symptoms.replace(' ', '').split(',')]
    for value in inputs:
        if value.strip() not in symptom_list:
            return "ຂໍອະໄພ ອາການທີ່ປ້ອນມາບໍ່ມີໃນລະບົບບໍ່ສາມາດວິນິໄສໄດ້"
    return True

def validate_inputs(age, weight, blood_pressure, temperature, symptoms):

    if np.isnan(age) or np.isnan(weight) or np.isnan(blood_pressure) or np.isnan(temperature) or not symptoms:
        return "ກະລຸນາໃສ່ຂໍ້ມູນໃຫ້ຄົບຖ້ວນ"
    if  weight <= 0:
        return "ກະລຸນາໃສ່ຂໍ້ມູນນ້ຳໜັກໃຫ້ຖືກຕ້ອງ"
    if  blood_pressure <= 60:
        return "ຄວາມດັນບໍ່ຖືກຕ້ອງຕ່ຳເກີນໄປ ແນະນຳໃຫ້ໄປພົບແພດໂດຍດ່ວນ"
    if  blood_pressure >= 200:
        return "ຄວາມດັນບໍ່ຖືກຕ້ອງສູງເກີນໄປ ແນະນຳໃຫ້ໄປພົບແພດໂດຍດ່ວນ"
    if temperature < 35:
        return "ອຸນຫະພູມຮ່າງກາຍຕ່ຳເກີນໄປບໍ່ປົກກະຕິ ແນະນຳໃຫ້ໄປພົບແພດໂດຍດ່ວນ"
    if temperature > 42:
        return "ອຸນຫະພູມຮ່າງກາຍສູງເກີນໄປບໍ່ປົກກະຕິ ແນະນຳໃຫ້ໄປພົບແພດໂດຍດ່ວນ"
    return None

#Interface
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@300;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, stWrite {
        font-family: 'Noto Sans Lao', sans-serif !important;
        font-weight: 300;
        font-size: 24px;
    }
    button {
        width: 150px;
        height: 70px;
        padding-top: 5px !important;
        padding-bottom: 5px !important;
    }
    .stNumberInput label {
        font-size: 20px !important; /* Adjust the font size as needed */
        font-family: 'Noto Sans Lao', sans-serif !important;
        font-weight: 300;
    }
    .detail {
        padding: 30px 0 0 0;
        font-size: 32px !important; /* Font size for warnings */
        font-family: 'Noto Sans Lao', sans-serif !important;
        font-weight: 600 !important;
    }
    .result {
        color: #0000FF;
        font-size: 30px !important; /* Font size for warnings */
        font-family: 'Noto Sans Lao', sans-serif !important;
        font-weight: 600 !important;
            }
    .result2{
        font-size: 26px !important; /* Font size for warnings */    
        font-family: 'Noto sans lao', sans-serif;
        font-weight: 600;
        margin-left; 16px
            }
    .warning {
        color: #FF0000;
        font-size: 24px !important; /* Font size for warnings */
        font-family: 'Noto Sans Lao', sans-serif !important;
        font-weight: 600 !important;
    }
    
    .custom-warning {
    color: red !important; /* Warning color */
    font-size: 20px !important; /* Font size for warnings */
    font-family: 'Noto Sans Lao', sans-serif !important;
    font-weight: 600 !important;
    }
    .tab-space {
        display: inline-block;
        width: 2em; /* Adjust the width as needed */
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app()
buf1, col4 = st.columns([1, 5])
with col4:
    st.image("FSN_Header.png", width=130)

buf2, col3 = st.columns([1.4, 3])
with col3:
    st.markdown("<h1 style='margin-bottom: 20px;padding:0 0 20px 0'>Disease prediction based on symptoms</h1>", unsafe_allow_html=True)



buff, col1, buff, col2, buff= st.columns([1,2,0.2,2.6,1])

with col1:
    st.markdown('## Enter Patient Details')
    # Collect user input
    gender = st.radio(r"$\textsf{\Large ເພດ}$", ('Male', 'Female'))
    age = st.number_input(r"$\textsf{\Large ອາຍຸ}$",min_value=1)
    weight = st.number_input(r"$\textsf{\Large ນ້ຳໜັກ}$", step=1)
    blood_pressure = st.number_input(r"$\textsf{\Large ຄວາມດັນ}$", step=1)
    temperature = st.number_input(r"$\textsf{\Large ອຸນຫະພູມ}$", step=1.)
    symptoms = st.multiselect(r"$\textsf{\Large ອາການ}$", symptom_list )
    symptoms = ', '.join(symptoms)

    # Display the input values
    st.markdown('## Input Values:')
    st.write(f'### ເພດ: {gender}')
    st.write(f'### ອາຍຸ: {age} ປີ')
    st.write(f'### ນ້ຳໜັກ: {weight} Kg')
    st.write(f'### ຄວາມດັນ: {blood_pressure}')
    st.write(f'### ອຸນຫະພູມ : {temperature} °C')
    st.write(f'### ອາການ: {symptoms}')

    if st.button('Predict'):
        validation_error = validate_inputs(age, weight, blood_pressure, temperature, symptoms)
        if validation_error:
            st.write(f'<p class="custom-warning">⚠️ {validation_error}</p>', unsafe_allow_html=True)

        else:    
            symptom_check = is_symptom(symptoms, symptom_list)
            if symptom_check == True:
                 top_3_classes, top_3_probabilities = get_predict()
            with col2:
                st.markdown(f'<p class="result" style="color:#000000; padding:14px 0 0 0">ຜົນການບົ່ງມະຕິພະຍາດ:</p>', unsafe_allow_html=True)
                st.write('<p class="warning">ໝາຍເຫດ: !ຜົນການບົ່ງມະຕິພະຍາດນີ້ ເປັນພຽງຜົນບົ່ງມະຕິເບື້ອງຕົ້ນເທົ່ານັ້ນ!</p>', unsafe_allow_html=True)

                st.write(f'<p class="result">{get_info(0).get("disease name")} <span class="tab-space"></span>ດ້ວຍຄ່າ Probability: {top_3_probabilities[0]*100:.2f}%</p>', unsafe_allow_html=True)
                st.write(f'<p class="result2" style="color: #0080FF">{get_info(1).get("disease name")} &nbsp;&nbsp;&nbsp;&nbsp;ດ້ວຍຄ່າ Probability: {top_3_probabilities[1]*100:.2f}%</p>', unsafe_allow_html=True)
                st.write(f'<p class="result2" style="color: #66B2FF">{get_info(2).get("disease name")} &nbsp;&nbsp;&nbsp;&nbsp;ດ້ວຍຄ່າ Probability: {top_3_probabilities[2]*100:.2f}%</p>', unsafe_allow_html=True)
                
    
                st.write(f'<p class="detail">ຂໍ້ມູນຂອງພະຍາດ: {get_info(0).get("disease name")}</p>', unsafe_allow_html=True)
                st.write(f'<p class="result2">{get_info(0).get("Description", "No description available")}</p>', unsafe_allow_html=True)

                st.write('<p class="detail">ການຮັກສາເບື້ອງຕົ້ນ:</p>', unsafe_allow_html=True)
                st.write(f'<p class="result2">{get_info(0).get("Treatmenting", "No treatment information available")}<h4>', unsafe_allow_html=True)


