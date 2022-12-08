#import pickle
#from pathlib import Path 

import streamlit as st #pip install streamlit 
from sklearn import datasets 
from sklearn.svm import SVC #support vector machine algorithm
from sklearn.neighbors import KNeighborsClassifier #KNN algorithm
from sklearn.naive_bayes import GaussianNB #naive bayes algorithm
from sklearn.ensemble import RandomForestClassifier #random forest algorithm
import streamlit_authenticator as stauth #pip install streamlit-authenticator
import streamlit_extras as stext #pip install streamlit-extras
from streamlit_extras.app_logo import add_logo #adding logo
from streamlit_extras.add_vertical_space import add_vertical_space #adding space vertically
import pandas as pd #pip install pandas openpyxL
import numpy as np 


from streamlit_option_menu import option_menu
from PIL import Image

# --- USER AUTHENTICATION --- #
#names = ["Daffa Pratama", "Novelia Agatha Santoso", "Nurul Fadillah"]
#usernames = ["daffa.pratama", "novelia.santoso", "nurul.fadillah"]

#file_path = Path(__file__).parent / "hashed_pw.pkl"
#with file_path.open("rb") as file:
#    hashed_passwords = pickle.load(file)

#authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
#    "some_cookie_name", "some_signature_key", cookie_expiry_days=100)

#name, authenticaion_status, username = authenticator.login("Login", "main") 

#if authenticaion_status == False:
#    st.error("Username/password salah")

#if authenticaion_status == None:
#    st.warning("Masukkan username dan password")

#if authenticaion_status:
    
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

#    authenticator.logout("Logout", "sidebar")

selected = option_menu(
            menu_title=None,
            options=["Home", "Prediksi", "Visualisasi"],
            icons=["house-fill", "diagram-3-fill", "bar-chart-fill"],
            default_index=0,
            orientation="horizontal",
            styles={
                "icon": {"color": "black", "font-size": "25px"},
                "nav-link": {
                    "--hover-color": "#c45c07"
                },
                "nav-link-selected": {
                    "background-color": "#960018",
                },
            }
        )

if selected == "Home":
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
            (0.1, 2, 0.2, 1, 0.1)
        )
        row0_1.title("Selamat Datang di Web Prediksi Karyawan Resign")

        with row0_2: add_vertical_space()
        row0_2.subheader("PT. HIT X BINUS University Project")

if selected == "Prediksi":
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
            (0.1, 2, 0.2, 1, 0.1)
        )
        row0_1.title("Prediksi Karyawan Resign")

        with row0_2: add_vertical_space()
        row0_2.subheader("Download Template CSV")
        text_contents = '''
        NIK, NAMA, PERSONAL_SUB_AREA, EMPLOYEE_STATUS, EMPLOYEE_LEVEL, DEPARTEMEN, MARITAL_STATUS, AGE, DAILY_HOUR, YEARS_IN_COMPANY, GENDER, JARAK, WAKTU_TEMPUH, LABEL
        '''
        row0_2.download_button('Download Template', text_contents, 'Template Prediksi.csv', 'text/csv')

        data_file=st.file_uploader("Upload file karyawan yang ingin digenerate", type=["csv"])
        if data_file is not None:
            st.write(type(data_file))
            file_details = {"filename":data_file.name,
            "filetype":data_file.type, "filesize":data_file.size}
            df = pd.read_csv(data_file)
            st.dataframe(df)

if selected == "Visualisasi":
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
            (0.1, 2, 0.2, 1, 0.1)
        )
        row0_1.title("Visualisasi Data Karyawan PT. HIT")

        with row0_2: add_vertical_space()
        row0_2.image('https://www.braderian.id/wp-content/uploads/2020/11/braderian-polytron-logo.png', width=100) 
        row0_2.caption('in collaboration with')
        row0_2.image('https://sis.binus.ac.id/wp-content/uploads/2015/08/logo-SIS-PNG.png', width=100)

st.caption("PT. Hartono Istana Teknologi X BINUS University")
