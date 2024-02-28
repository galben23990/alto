import streamlit as st
from pages import Calculate_Holdings
from pages import Create_Investor_PNL
import time
from datetime import date

def main():
    st.set_page_config(layout="wide")
    # Create tabs
    tab1,tab2 = st.tabs(["Calculate Holdings", "Create Investor PNL"])

    # Define the content of each tab
    with tab1:
        Calculate_Holdings.Calculate_Holdings()
    with tab2:
        Create_Investor_PNL.Create_Investor_PNL()



# Entry point of the application
if __name__ == "__main__":
    main()
