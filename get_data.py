import pandas as pd
import time  # required to call any time commands (i.e. delays)
from google.cloud import bigquery
import pickle
import ctypes
import numpy as np
import json
from google.oauth2 import service_account
import pygsheets
import json
import os
from datetime import datetime
import streamlit as st
import json
import pygsheets

import json
import pygsheets
import streamlit as st

# Load the service account credentials from Streamlit secrets
service_account_info = st.secrets["FIREBASE_SERVICE_ACCOUNT"]

# Convert AttrDict to a regular dictionary
service_account_dict = dict(service_account_info)

# Write the credentials to a temporary file
temp_cred_file = "temp_service_account.json"
with open(temp_cred_file, 'w') as file:
    json.dump(service_account_dict, file)

# Authorize pygsheets with the service account
gc = pygsheets.authorize(service_file=temp_cred_file)


def get_data():
    sh = gc.open("Alto")
    wks = sh.worksheet_by_title("payment")
    payment = wks.get_as_df()
    payment.payment_date=pd.to_datetime(payment.payment_date)
    wks = sh.worksheet_by_title("commitment")
    commitment = wks.get_as_df()
    commitment.commitment_date = pd.to_datetime(commitment.commitment_date)
    wks = sh.worksheet_by_title("report")
    report = wks.get_as_df()
    return payment,commitment,report

def holding_per_date(payment,d):
    dt = datetime.combine(d, datetime.min.time())
    payment = payment[payment.payment_date <= dt]
    payment.groupby(by=['investor_ID', 'investor_name', 'feeder'])['payment_sum'].sum()
    holdings = (payment.groupby(by=['investor_ID', 'investor_name', 'feeder'])['payment_sum'].sum())
    holdings = holdings.reset_index()
    holdings["holdings"] = holdings['payment_sum'] / holdings['payment_sum'].sum()
    return holdings
if __name__ == "__main__":
    get_data()
# all_creds = []
# for i, row in cred_df.iterrows():
#     # if (row.latest_date_uploaded=='') or (datetime.datetime.fromisoformat(row.latest_date_uploaded)<datetime.datetime.now()-datetime.timedelta(5)):
#     tmp_dict = row.to_dict()
#     all_creds.append(tmp_dict)
# dict_cf = {}
# what is a freeze reuqriment.txt terimail command?
