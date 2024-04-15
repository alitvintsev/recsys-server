import json
import requests
import folium
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium


def get_prediction(text, ip='127.0.0.1'):
    # params = {'text': text}
    if text:
        url = f'http://{ip}:8000/predict/{text}'
        r = requests.get(url=url)
        # r.encoding = 'utf-8'
        result = r.json()
        res_table = pd.DataFrame().from_records(eval(result))
        st.table(res_table)


def get_points_data(path):
    pass


def run_app():
    # headers
    st.title('Restaurant RecSys') 
    st.write("by Alexander Litvintsev")
    # st.image(f"static/img/sentiment_icon.jpeg", width=300)

    # get user input from text areas in a Streamlit app
    description = st.text_area(label="Input Description", value="", height=None)

    st.button(label="Get Prediction", on_click=get_prediction(description, ip='91.105.196.201'))

    # m = folium.Map(location=[39.949610, -75.150282], zoom_start=5)
    # folium.Marker(
    #     [39.949610, -75.150282],
    #     popup="Liberty Bell",
    #     tooltip="Liberty Bell"
    # ).add_to(m)

    # st_data = st_folium(m, width=725)


if __name__ == "__main__":
    run_app()


