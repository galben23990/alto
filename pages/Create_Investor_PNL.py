import streamlit as st
from datetime import date, datetime
import json
import pandas as pd
import numpy as np


def Create_Investor_PNL():
    def get_quarter_dates(quarter):
        quarters = {
            "Q1-23": date(2023, 3, 31),
            "Q2-23": date(2023, 6, 30),
            "Q3-23": date(2023, 9, 30),
            "Q4-22": date(2022, 12, 31),
        }
        return quarters.get(quarter)



    st.title('Investor P&L Report')
    with open(r'data\indexed _costumer_data.json') as f:
        data_costumer = json.load(f)

    with open(r'data\parameters.json') as f:
        parameters = json.load(f)



    costumers=data_costumer.keys()
    parameters=parameters.keys()


    # Using columns for layout
    col1, col2,col3= st.columns([1,1,1])

    with col1:
        st.header('Select Quarter End')
        quarter = st.selectbox('Choose Quarter For Report', ['Q1-23', 'Q2-23', 'Q3-23', 'Q4-22'])
        end_date = get_quarter_dates(quarter)

    with col2:
        st.header('Select Costumer')
        costumer = st.selectbox('Choose Costumer', costumers)
    with col3:
        st.header('Select Parameters')
        parameter = st.multiselect('Choose Parameters', parameters)


    submit = st.button('Submit Report')

    # Placeholder for output
    if submit:
        # Your calculation logic goes here
        st.write('Calculations for selected dates...')
        st.write(f'Calculating P&L from {start_date} to {end_date}...')


