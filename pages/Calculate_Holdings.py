import copy
import streamlit as st
from datetime import date, datetime
from get_data import get_data,holding_per_date
import numpy as np
import matplotlib.pyplot as plt
import time
def get_pie_style(labels, sizes, col):
    explode = np.array([s / np.array(sizes).sum() for s in sizes])
    fig1, ax1 = plt.subplots()
    a, b, c = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode=explode, shadow=True)
    # _, _, autotexts = pie(fbfrac, labels=fblabel, autopct='%1.1f%%', pctdistance=0.8, startangle=90, colors=fbcolor)
    for autotext in b:
        autotext.set_color('white')
    for autotext in c:
        autotext.set_color('white')
    # Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0, 0), 0.70, fc='gray', alpha=0.0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax1.axis('equal')
    ax1.set_title(col.replace('_', ' '), color='white')
    ax1.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    return fig
def Calculate_Holdings():
    def get_quarter_dates(quarter):
        current_year = date.today().year
        quarters = {
            "Q1-23": date(2023, 3, 31),
            "Q2-23": date(2023, 6, 30),
            "Q3-23":  date(2023, 9, 30),
            "Q4-22": date(2022, 12, 31),
        }
        return quarters.get(quarter, (None, None))

    st.title('Calculate Holdings')

    # Using columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.header('Select Quarter or Date')
        quarter = st.selectbox('Choose Quarter or Date', ['','Q1-23', 'Q2-23', 'Q3-23', 'Q4-22','Specific Date'],)
        if quarter!='' and quarter!='Specific Date':
            d= get_quarter_dates(quarter)
        else:
            d=datetime.today()

    if quarter=='Specific Date':
        with col2:
            st.header('Specific Date')
            d = st.date_input("Enter Specific Date", datetime(2024, 1, 1))

    submit = st.button('Submit')

    # Placeholder for output
    if submit:
        spinner_load = st.empty()
        expander_load = st.empty()
        with spinner_load.container(border=True):
            with st.spinner("Calculating Holdings..."):
                col3, col4 = st.columns(2)
                with col3:
                    # Perform calculations and display results
                    payment, commitment, report = get_data()
                    holdings = holding_per_date(payment, d)
                    holdings = holdings.sort_values(by='holdings', ascending=False)
                    holdings = holdings.set_index("investor_ID")
                    holdings['holdings_percent'] = holdings['holdings'].apply(lambda x: f'{x * 100:.2f}%')
                    st.dataframe(holdings[['investor_name', 'feeder', 'holdings_percent']])
                with col4:
                    top_holdings = holdings.iloc[:7]
                    labels = top_holdings['investor_name'].to_list()
                    sizes = top_holdings["holdings"].to_list()
                    others_size = 1 - np.array(sizes).sum()
                    if others_size > 0:
                        labels.append('others')
                        sizes.append(others_size)
                    fig = get_pie_style(labels, sizes, "Holdings")
                    st.pyplot(fig)

