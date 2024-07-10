import streamlit as st
from datetime import date, datetime
import pandas as pd
from get_data import matix_per_date_including_openning_movement

def Calculate_Matrix():
    def get_quarter_dates(quarter):
        current_year = date.today().year
        quarters = {
            "Q1-24": date(2024, 3, 31),
            "Q2-23": date(2023, 6, 30),
            "Q3-23": date(2023, 9, 30),
            "Q4-23": date(2023, 12, 31),
        }
        return quarters.get(quarter, (None, None))

    st.title('Calculate Matrix')
    # Using columns for layout
    col1, col2 = st.columns(2)
    with col1:
        st.header('Select Quarter or Date')
        quarter = st.selectbox('Choose Quarter or Specific Date', ['', 'Q1-24', 'Q2-23', 'Q3-23', 'Q4-23', 'Specific Date'])
        if quarter != '' and quarter != 'Specific Date':
            d = get_quarter_dates(quarter)
        else:
            d = datetime.today()

    if quarter == 'Specific Date':
        with col2:
            st.header('Specific Date')
            d = st.date_input("Enter Specific Date", datetime(2024, 1, 1))

    submit = st.button('Run')
    if submit:
        spinner_load = st.empty()
        expander_load = st.empty()
        with spinner_load.container():
            with st.spinner("Calculating Matrix..."):
                # This section is adjusted to properly display the DataFrame in Streamlit
                df = matix_per_date_including_openning_movement(datetime(d.year, d.month, d.day))
                st.dataframe(df)
                temp_excel_file = "matrix_data.xlsx"
                df.to_excel(temp_excel_file, index=False)
                # Provide download link
                with open(temp_excel_file, "rb") as file:
                    st.download_button(
                        label="Download data as Excel",
                        data=file,
                        file_name=temp_excel_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

def main():
    Calculate_Matrix()

if __name__ == "__main__":
    main()
