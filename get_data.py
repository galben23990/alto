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



# key_dict= json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
# cred = service_account.Credentials.from_service_account_info(key_dict)
# client =bigquery.Client.from_service_account_info(key_dict)

# Load the service account credentials
service_account_info = json.loads(st.secrets["FIREBASE_SERVICE_ACCOUNT"]["json"])

# Authorize pygsheets with the service account
gc = pygsheets.authorize(service_account_info=service_account_info)# key_dict= json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
#

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
