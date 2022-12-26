import streamlit as st #pip install streamlit 
from streamlit_option_menu import option_menu #option menu streamlit extras
import streamlit_authenticator as stauth #pip install streamlit-authenticator
import streamlit_extras as stext #pip install streamlit-extras
from streamlit_extras.app_logo import add_logo #adding logo
from streamlit_extras.add_vertical_space import add_vertical_space #adding space vertically

from sklearn import datasets 
from sklearn import metrics
from sklearn.metrics import (confusion_matrix,accuracy_score,matthews_corrcoef,precision_score,recall_score,f1_score)
from sklearn.ensemble import RandomForestClassifier #random forest algorithm
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split #split data train and data test

import pandas as pd #pip install pandas openpyxL
import yaml
from yaml import SafeLoader
import numpy as np 
from PIL import Image

# --- BAKGROUND DESIGN --- #

st.set_page_config(page_title="Website Prediksi Karyawan",layout="wide")

page_bg_img = """
    <style> 
    [data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1588345921523-c2dcdb7f1dcd?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80");
    background-size: cover;
    }

    [data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
    }
    </style>
    """
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- NAVIGATION DESIGN --- #

selected = option_menu(
            menu_title=None,
            options=["Home", "Prediksi"],
            icons=["house-fill", "diagram-3-fill"],
            default_index=0,
            orientation="horizontal",
            styles={
                "icon": {"color": "black", "font-size": "25px"},
                "nav-link": {
                    "--hover-color": "#808080"
                },
                "nav-link-selected": {
                    "background-color": "#960018",
                },
            }
        )

# -- HOME PAGE -- #

if selected == "Home":
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
            (0.1, 2, 0.2, 1, 0.1)
        )
        row0_1.title("Selamat Datang di Web Prediksi Karyawan Resign")

        with row0_2: add_vertical_space()
        row0_2.subheader("PT. HIT X BINUS University Project")
        row0_2.image('https://www.braderian.id/wp-content/uploads/2020/11/braderian-polytron-logo.png', width=100) 
        row0_2.caption('in collaboration with')
        row0_2.image('https://sis.binus.ac.id/wp-content/uploads/2015/08/logo-SIS-PNG.png', width=100)

# -- PREDICTION PAGE -- #

if selected == "Prediksi":
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
            (0.1, 2, 0.2, 1, 0.1)
        )
        row0_1.title("Prediksi Karyawan Resign")

        with row0_2: add_vertical_space()
        row0_2.subheader("Download Template CSV")
        text_contents = '''
        NIK, NAME, GENDER, AGE, EDUCATION_LEVEL_ID, MARITAL_STATUS, DEPARTMENT, PERSONAL_SUB_AREA, EMPLOYEE_STATUS, EMPLOYEE_LEVEL, DAILY_HOUR, YEARS_IN_COMPANY, DISTANCE, DURATION, LABEL
        '''
        row0_2.download_button('Download Template', text_contents, 'Template Prediksi.csv', 'text/csv')

        # -- FUNGSI CONVERT DAILY HOUR MENJADI SECONDS -- #

        def to_seconds(s):
            hr, minutes, sec = [float(x) for x in s.split(':')]
            return hr*3600 + minutes*60 + sec

        # -- FUNGSI DATA CLEANING -- #

        def clean_data_automated(datasets):
            datasets = datasets.dropna(axis=0)
            datasets.Daily_Hour = datasets.Daily_Hour.apply(to_seconds)
            datasets.Daily_Hour = datasets.Daily_Hour / 3600
            datasets.Distance = datasets.Distance / 1000
            datasets.Duration = datasets.Duration / 60
            datasets.Age = datasets.Age.astype('int64')
            datasets.Years_in_Company = datasets.Years_in_Company.astype('int64')
            datasets_clean = datasets.drop(columns = ['NIK','Name','Personal_Sub_Area'])
            datasets_clean.Marital_Status = datasets_clean.Marital_Status.map({'Lajang':1, 'Menikah':2, 'Cerai Mati':3, 'Cerai Hidup':3, 'Duda/Janda':4})
            datasets_clean.Employee_Level = datasets_clean.Employee_Level.map({'Level 1':1, 'Level 2**':2, 'Level 2*':3, 'Level 3**':4, 'Level 3*': 5, 'Level 4**':6, 'Level 4*':7, 'Level 5**': 8, 'Level 5*':9, 'Level 6': 10, 'Level 7':11})
            datasets_clean.Employee_Status = datasets_clean.Employee_Status.map({'Magang':1, 'Kontrak pertama':2, 'Kontrak Perpanjangan':3, 'Kontrak pembaharuan':3, 'Tetap':4})
            datasets_clean.Gender = datasets_clean.Gender.map({'Laki-laki':1, 'Perempuan':2})
            datasets_clean.Education_Level_Id = datasets_clean.Education_Level_Id.astype('int64')
            datasets_clean.Departemen = datasets_clean.Departemen.map({'Executive':1,'FIRA':2,'Cost Control': 3, 'Testing Laboratory':4, 'Procurement':5, 'Administration':6,'Research and Development': 7, 'Finance': 8,'Commercial':9,'Production':10})
            datasets_clean.Label = datasets_clean.Label.dropna(axis=0)
            datasets_clean.Label = datasets_clean.Label.map({'actives':0,'terminates':1})
            return datasets_clean

        data_file=st.file_uploader("Upload file karyawan yang ingin digenerate", type=["csv"])
        
        # -- MEMBACA KONDISI FILE SUDAH ADA UPLOAD DARI USER ATAU BELUM -- #
        
        if data_file is not None:
            df = pd.read_csv(data_file)
            data_clean = clean_data_automated(df)

            # -- MAKE DATAFRAME -- #

            st.header("Data Frame dari Dataset")
            st.dataframe(df)
            st.header("Data Frame dari Data Cleaning")
            st.dataframe(data_clean, use_container_width=True) #framing data clean

            # -- CONVERT DATA TO ARRAY -- #

            X=data_clean.drop(columns=['Label'], axis=1) #Menentuan Features, ':' berarti memilih semua baris, dan ':-1' mengabaikan kolom terakhir
            y=data_clean['Label'] #Menentukan Label, ':' berarti memilih semua baris, dan '-1:' mengabaikan semua kolom kecuali kolom terakhir

            # -- SET DATA TRAIN AND DATA TEST -- #

            X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=42) #Mengambil 75% untuk Data Training

            # -- CLASSIFIER -- #

            clf=RandomForestClassifier(n_estimators=1000) #Membuat model
            clf.fit(X_train,y_train) #Training model yang dibuat
            ypredict=clf.predict(X)

            # -- ACCURACY & CONFUSION MATRIX & PRECISION & RECALL & F1-SCORE -- #

            kol1, kol2, kol3, kol4, kol5 = st.columns(5, gap="small")

            with kol1 :
                st.subheader("Confusion Matrix")
                st.text(confusion_matrix(y,ypredict))
            
            with kol2 :
                st.subheader("Accuracy")
                st.text(accuracy_score(y,ypredict)*100)
            
            with kol3 :
                st.subheader("Precision")
                st.text(precision_score(y,ypredict))
        
            with kol4 : 
                st.subheader("Recall")
                st.text(recall_score(y,ypredict))
        
            with kol5 :
                st.subheader("F1-Score")
                st.text(f1_score(y,ypredict))
            
            # -- INPUT USER NEW TESTING DATA -- #
            
        #else:
            with st.form('newinput'):
                st.subheader("Input value yang ingin diprediksi :")
                age = st.slider('Age', 20, 80)
                gender = st.selectbox('Gender', ('Laki-laki', 'Perempuan'))
                departemen = st.selectbox('Department', ('Administration','Commercial','Cost Center','Executive','Finance','FIRA','Procurement','Production','Research and Development','Testing Laboratory'))
                marital_status = st.selectbox('Marital Status', ('Lajang','Menikah','Cerai Hidup','Cerai Mati','Duda/Janda'))
                educational_level_id = st.selectbox('Pendidikan Terakhir', ('SD','SMP','SMA','SMK','Diploma','S1','S2','Post/Grad Diploma','S3'))
                employee_status = st.selectbox('Status Karyawan', ('Magang','Kontrak Pertama','Kontrak Perpanjangan','Kontrak Pembaharuan','Tetap'))
                employee_level = st.selectbox('Level Karyawan', ('Level 1','Level 2**','Level 2*','Level 3**','Level 3*','Level 4**','Level 4*','Level 5**','Level 5*','Level 6','Level 7'))
                daily_hour = st.slider('Daily Hours', 6.00, 12.00)
                years_in_company = st.slider('Years in Company', 0, 30)
                distance = st.slider('Jarak', 0.00, 100.00)
                duration = st.slider('Durasi', 0.00, 100.00)
                
                datalist = {
                    "Education_Level_Id" : [educational_level_id],
                    "Employee_Status": [employee_status],
                    "Employee_Level": [employee_level],
                    "Departemen": [departemen],
                    "Marital_Status" : [marital_status],
                    "Age": [age],
                    "Daily_Hour": [daily_hour],
                    "Years_in_Company": [years_in_company],
                    "Gender": [gender],
                    "Distance": [distance],
                    "Duration": [duration]
                }

                # -- MENGUBAH DATALIST DICTIONARY MENJADI DATAFRAME -- #

                dataframe = pd.DataFrame(data=datalist)

                # -- TOMBOL SUBMIT INPUT USER -- #

                submitted = st.form_submit_button('Submit')
                if submitted is True :
                    st.success('Submit Berhasil')
                    
                    # -- MENAMPILKAN INPUT YANG DILAKUKAN USER -- #

                    st.subheader('Display Input Value User')
                    st.dataframe(dataframe, use_container_width=True)
                
                    def clean_data_input(dataframe):
                        dataframe = dataframe
                        dataframe.Marital_Status = dataframe.Marital_Status.map({'Lajang':1, 'Menikah':2, 'Cerai Mati':3, 'Cerai Hidup':3, 'Duda/Janda':4})
                        dataframe.Employee_Level = dataframe.Employee_Level.map({'Level 1':1, 'Level 2**':2, 'Level 2*':3, 'Level 3**':4, 'Level 3*': 5, 'Level 4**':6, 'Level 4*':7, 'Level 5**': 8, 'Level 5*':9, 'Level 6': 10, 'Level 7':11})
                        dataframe.Employee_Status = dataframe.Employee_Status.map({'Magang':1, 'Kontrak Pertama':2, 'Kontrak Perpanjangan':3, 'Kontrak Pembaharuan':3, 'Tetap':4})
                        dataframe.Gender = dataframe.Gender.map({'Laki-laki':1, 'Perempuan':2})
                        dataframe.Education_Level_Id = dataframe.Education_Level_Id.map({'SD':1, 'SMP':2, 'SMA':3, 'SMK':4, 'Diploma':5, 'S1':6, 'S2':8, 'Post/Grad Diploma':9,'S3':10})
                        dataframe.Departemen = dataframe.Departemen.map({'Executive':1,'FIRA':2,'Cost Control': 3, 'Testing Laboratory':4, 'Procurement':5, 'Administration':6,'Research and Development': 7, 'Finance': 8,'Commercial':9,'Production':10})
                        return dataframe
                    
                    # -- MENAMPILKAN DATA CLEANING DARI INPUT USER -- #

                    newdata = clean_data_input(dataframe)
                    st.subheader('Display Input Value Data Clean')
                    st.dataframe(newdata, use_container_width=True)

                    # -- MEMPREDIKSI INPUTAN USER -- #
                    
                    ypredict = clf.predict_proba(newdata)
                    prediction = list(map(round,ypredict))
                    
                    st.subheader('Prediksi Karyawan Tersebut Resign Adalah')
                    st.text(ypredict)

                    
