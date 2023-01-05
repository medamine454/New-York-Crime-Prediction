import pandas as pd
import folium
import streamlit as st
from folium.plugins import Draw
import os
from streamlit_folium import st_folium

from utils.preprocess import GetDummies
import utils.model as model_utils


def show_result(msg, coords, m):
    return folium.Marker(location=coords, popup='<i>Test</i>',
                         icon=folium.DivIcon(
                             html=f"""<div style="font-family: Century Gothic; color: red; background-color:white">{msg}</div>""")
                         ).add_to(m)


st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 375px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)
st.title('New York Crime Prediction ')
with st.sidebar:
    st.title(' NEW YORK Crimes prediction: USER GUIDE ')
    st.write('* Pick the date, gender, race, destination, Boro,and destination type  ')
    st.write('* In order to get the prediction result, please put a marker on the map  ')

col1, col2 = st.columns([3, 1])

# Import the required library

with col1:
    st.subheader("Map :")

    m = folium.Map(location=[40.730610, -73.935242], zoom_start=10)
    Draw(export=True).add_to(m)
    folium.Marker(location=[39.949610, -75.150282], popup='<i>Test</i>').add_to(m)

    output = st_folium(m, width=700, height=500)

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.horizontal = False
with col2:
    st.subheader("Enter your  data")

    gender = st.radio(
        " What\'s your gender",
        ('Male ', 'Female', 'UNKNOWN'))
    date = st.date_input("pick a date", value=None, min_value=None, max_value=None, key=None)
    hour = st.slider("Hour:", min_value=0, max_value=24)
    age = st.slider('Pick an age', 0, 100, 6)
    st.write("I'm ", age, 'years old')
    race = st.selectbox(
        'Pick your race ',
        ('WHITE', 'WHITE HISPANIC', 'BLACK', 'UNKNOWN', 'ASIAN / PACIFIC ISLANDER'
                                                        'BLACK HISPANIC', 'AMERICAN INDIAN/ALASKAN NATIVE', 'OTHER'))

    place = st.radio("Place:", ('BROOKLYN', 'STATEN ISLAND', 'BRONX', 'QUEENS', 'MANHATTAN', 'UNKNOWS'))
    place_type = st.selectbox( 'Pick your destination type ',('COMMERCIAL_BUILDING','Park_Street','Residence_House'))

# gender input
if gender == 'Male':
    mgender = 'M'
if gender == 'Female':
    mgender = 'F'
else:
    mgender = 'UNKNOWN'

# age input
if age in [25, 44]:
    mage = '25-44'

if age in [45, 64]:
    mage = '45-64'
if age in [18, 24]:
    mage = '18-24'

if age < 18:
    mage = '<18'

if age > 65:
    mage = '65+'

else:
    mage = 'UNKNOWN'

day = date.day
month = date.month
weekday = date.isoweekday()

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

weekday_index = date.isoweekday()
weekday = WEEKDAYS[weekday_index - 1]

try:
    coord = output["last_active_drawing"]["geometry"]["coordinates"]
    # print(coord)
    details = {'hour': hour, 'ADDR_PCT_CD': 41., 'month': month, 'day': day, 'Latitude': coord[1],
               'Longitude': coord[0],
               'BORO_NM': place, "WEEKDAY": weekday, 'VIC_AGE_GROUP': mage, 'VIC_RACE': race, 'VIC_SEX': mgender, 'PREM_TYP_DESC':place_type}
    df = pd.DataFrame(details, index=[0])

    dfs = pd.get_dummies(df)

    reference_df_path = "data/lgbm_reference_data.csv"
    prepare_data = GetDummies(reference_df_path)

    model_path = "data/xgb_clf.pkl"

    xgboost = model_utils.Model(model_path, "xgboost")
    # cols_when_model_builds = xgboost.model.get_booster().feature_names
    # cols_when_model_builds = ['ADDR_PCT_CD', 'month', 'day', 'hour', 'Latitude', 'Longitude', 'BORO_NM_BRONX', 'BORO_NM_BROOKLYN', 'BORO_NM_MANHATTAN', 'BORO_NM_QUEENS', 'BORO_NM_STATEN_ISLAND', 'BORO_NM_UNKNOWN', 'weekday_Friday', 'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday', 'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday', 'VIC_AGE_GROUP_18-24', 'VIC_AGE_GROUP_25-44', 'VIC_AGE_GROUP_45-64', 'VIC_AGE_GROUP_65+', 'VIC_AGE_GROUP_18', 'VIC_AGE_GROUP_UNKNOWN', 'VIC_RACE_AMERICAN_INDIAN/ALASKAN_NATIVE', 'VIC_RACE_ASIAN_/_PACIFIC_ISLANDER', 'VIC_RACE_BLACK', 'VIC_RACE_BLACK_HISPANIC', 'VIC_RACE_OTHER VIC_RACE_UNKNOWN', 'VIC_RACE_WHITE', 'VIC_RACE_WHITE_HISPANIC', 'VIC_SEX_F VIC_SEX_M', 'VIC_SEX_UNKNOWN', 'PREM_TYP_DESC_COMMERCIAL_BUILDING', 'PREM_TYP_DESC_Park_Street', 'PREM_TYP_DESC_Residence_House', 'PREM_TYP_DESC_COMMERCIAL BUILDING', 'PREM_TYP_DESC_Park_Street', 'PREM_TYP_DESC_Residence_House']
    model_path = "data/LightGBM.txt"
    lgbm = model_utils.Model(model_path, "lgbm")
    cols_when_model_builds = lgbm.model.feature_name()
    # print(details)


    dfs = prepare_data.transform(details, cols_when_model_builds)

    # result = model_utils.crime_name(xgboost.model.predict(dfs)[0])
    # folium.Marker(location=coord, popup='<i>Test</i>',
    #               icon=folium.DivIcon(
    #                   html=f"""<div style="font-family: Century Gothic; color: red; background-color:white">{result}</div>""")
    #               ).add_to(m)
    #st.write(lgbm.model.predict(dfs))

    # import pandas as pd
    # import numpy as np
    # result = lgbm.model.predict(dfs)
    # df = pd.DataFrame(
    #     result,
    #     columns=('MISDEMEANOR', "FELONY", "VIOLATION"))
    # #st.dataframe(df.style.apply(lambda x: "background-color: red"))
    #
    # st.table(df)
    # # print(lgbm.model.predict(dfs))

except:
    # st.write('nooooooooo')
    pass

# dictionary with list object in values


# details = {'hour': 0, 'ADDR_PCT_CD': 41.0, 'month': 12, 'day': 30, 'Latitude': 40.75766, 'Longitude': -74.198914,
#            'BORO_NM': 'BROOKLYN', 'WEEKDAY': 'Friday', 'VIC_AGE_GROUP': 'UNKNOWN', 'VIC_RACE': 'WHITE',
#            'VIC_SEX': 'UNKNOWN'}

with st.sidebar:
    st.title(' Prediction result ')
    st.write(' You are likely to be attacked by one of these crime type ')
    try:
        result = lgbm.model.predict(dfs)
        df = pd.DataFrame(
            result*100,
            columns=('MISDEMEANOR', "FELONY", "VIOLATION"))
        # st.dataframe(df.style.apply(lambda x: "background-color: red"))

        st.table(df)
    except:
        pass
