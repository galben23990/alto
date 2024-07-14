import streamlit as st
from pages import Calculate_Holdings
from pages import Calculate_Matrix
import time
from datetime import datea

def main():
    st.set_page_config(layout="wide")
    # Create tabs
    tab1,tab2 = st.tabs(["Calculate Holdings", "Calculate Matrix"])

    # Define the content of each tab
    with tab1:
        Calculate_Holdings.Calculate_Holdings()
    with tab2:
        Calculate_Matrix.Calculate_Matrix()



# Entry point of the application
if __name__ == "__main__":
    main()
