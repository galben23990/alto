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
    wks = sh.worksheet_by_title("distribution")
    distribution = wks.get_as_df()
    return payment, commitment, report, distribution


def holding_per_date(payment, d):
    dt = datetime.combine(d, datetime.min.time())
    payment = payment[payment.payment_date <= dt]
    holdings = (payment.groupby(by=['investor_ID', 'investor_name', 'feeder'])['payment_sum'].sum())
    holdings = holdings.reset_index()
    holdings["holdings"] = holdings['payment_sum'] / holdings['payment_sum'].sum()

    holdings_feeder = (payment.groupby(by=['feeder'])['payment_sum'].sum())
    holdings_feeder = holdings_feeder.reset_index()
    holdings_feeder.rename(columns={'payment_sum': 'feeder_holdings'}, inplace=True)
    holdings = pd.merge(holdings, holdings_feeder, on='feeder')
    holdings["holdings_per_feeder"] = holdings['payment_sum'] / holdings['feeder_holdings']
    return holdings


def disribution_make_table(distribution, payment):
    final_distribution = pd.DataFrame()
    pd.to_datetime(distribution.date)[0]
    for i, r in distribution.iterrows():
        tmp = holding_per_date(payment, pd.to_datetime(r.date))
        tmp.drop(columns=['payment_sum', 'holdings_per_feeder', 'feeder_holdings'], inplace=True)
        tmp.rename(columns={'holdings': 'distribution'}, inplace=True)
        tmp['date'] = r.date
        tmp['distribution'] = tmp['distribution'] * float(r['sum'].replace(',', ''))
        final_distribution = pd.concat([final_distribution, tmp])
    return final_distribution


def calc_interest(payment, distribution, cur_date, investment_part=0.7, intrest_fees=0.1):
    final_interest = []
    for id in payment.investor_ID.unique():
        CF_df = payment[payment.investor_ID == id]
        CF_df = CF_df[['payment_date', 'payment_sum']]
        CF_df.rename(columns={'payment_date': 'date', 'payment_sum': 'cash_flow'}, inplace=True)
        distribution_id = distribution[distribution.investor_ID == id]
        distribution_id = distribution_id[['date', 'distribution']]
        distribution_id.rename(columns={'distribution': 'cash_flow'}, inplace=True)
        distribution_id['date'] = pd.to_datetime(distribution_id['date'])
        CF_df = pd.concat([CF_df, distribution_id])
        CF_df = CF_df[CF_df.date <= cur_date]
        CF_df.loc[len(CF_df.index)] = [cur_date, 0.0]
        CF_df.sort_values(by='date', inplace=True)
        CF_df.reset_index(inplace=True, drop=True)
        CF_df['principal'] = 0
        CF_df['interest'] = 0
        CF_df['paid_interest'] = 0
        CF_df['paid_principal'] = 0
        for i, r in CF_df.iterrows():
            if i == 0:
                assert (r['cash_flow'] >= 0)
                print(r['cash_flow'])
                CF_df.loc[i, 'principal'] = r['cash_flow'] * investment_part
                CF_df.loc[i, 'interest'] = 0
                CF_df.loc[i, 'paid_interest'] = 0
                CF_df.loc[i, 'paid_principal'] = 0
            else:
                days = (r['date'] - CF_df.loc[i - 1, 'date']).days
                CF_df.loc[i, 'interest'] = CF_df.loc[i - 1, 'interest'] + (
                        CF_df.loc[i - 1, 'principal'] + CF_df.loc[i - 1, 'interest']) * (
                                                   1 - (1 + intrest_fees) ** (days / 365))
                CF_df.loc[i, 'paid_interest'] = CF_df.loc[i - 1, 'paid_interest']
                CF_df.loc[i, 'interest'] = CF_df.loc[i-1, 'interest'] + (CF_df.loc[i - 1, 'principal']+CF_df.loc[i-1, 'interest']) * ((1+intrest_fees)**(days/365)-1)
                CF_df.loc[i, 'paid_interest'] = CF_df.loc[i-1,'paid_interest']
                CF_df.loc[i, 'paid_principal'] = CF_df.loc[i - 1, 'paid_principal']
                if r.cash_flow > 0:
                    CF_df.loc[i, 'principal'] = CF_df.loc[i - 1, 'principal'] + r['cash_flow'] * investment_part
                else:
                    if -r['cash_flow'] < CF_df.loc[i, 'interest']:
                        CF_df.loc[i, 'interest'] = CF_df.loc[i, 'interest'] + r['cash_flow']
                        CF_df.loc[i, 'principal'] = CF_df.loc[i - 1, 'principal']
                        CF_df.loc[i, 'paid_interest'] = CF_df.loc[i, 'paid_interest'] - r['cash_flow']
                    else:
                        CF_df.loc[i, 'paid_interest'] = CF_df.loc[i, 'paid_interest'] + CF_df.loc[i, 'interest']
                        CF_df.loc[i, 'paid_principal'] = CF_df.loc[i, 'paid_principal'] - (
                                r['cash_flow'] + CF_df.loc[i, 'interest'])
                        CF_df.loc[i, 'principal'] = CF_df.loc[i - 1, 'principal'] + (
                                r['cash_flow'] + CF_df.loc[i, 'interest'])
                        CF_df.loc[i, 'interest'] = 0
        interest_id = pd.DataFrame(
            {'investor_ID': id, 'date': CF_df['date'].iloc[-1], 'interest': CF_df['interest'].iloc[-1],
             'principal': CF_df['principal'].iloc[-1], 'paid_interest': CF_df['paid_interest'].iloc[-1],
             'paid_principal': CF_df['paid_principal'].iloc[-1]}, index=[0])
        final_interest.append(interest_id)
    return pd.concat(final_interest)


def calc_sucsess_bound(payment, distribution, cur_date, investment_part=1, intrest_fees=0.07):
    final_interest = []
    for id in payment.investor_ID.unique():
        CF_df = payment[payment.investor_ID == id]
        CF_df = CF_df[['payment_date', 'payment_sum']]
        CF_df.rename(columns={'payment_date': 'date', 'payment_sum': 'cash_flow'}, inplace=True)
        distribution_id = distribution[distribution.investor_ID == id]
        distribution_id = distribution_id[['date', 'distribution']]
        distribution_id.rename(columns={'distribution': 'cash_flow'}, inplace=True)
        distribution_id['date'] = pd.to_datetime(distribution_id['date'])
        CF_df = pd.concat([CF_df, distribution_id])
        CF_df = CF_df[CF_df.date <= cur_date]
        CF_df.sort_values(by='date', inplace=True)
        CF_df.reset_index(inplace=True, drop=True)
        CF_df['lower_bound'] = 0

        days = (cur_date - CF_df['date'])
        # convert days to float number of days
        days = days.dt.total_seconds() / (24 * 60 * 60)
        bound = (CF_df['cash_flow'] * investment_part * (intrest_fees * (days / 365))).sum()
        bound_id = pd.DataFrame({'investor_ID': id, 'date': cur_date, 'bound': bound}, index=[0])
        final_interest.append(bound_id)
    return pd.concat(final_interest)


def matix_per_date(d, col_order=None):
    payment, commitment, report, distribution = get_data()
    distribution = disribution_make_table(distribution, payment)
    dt = datetime.combine(d, datetime.min.time())
    holdings = holding_per_date(payment, d)
    report = report.set_index("name").transpose()
    report.start_date = pd.to_datetime(report.start_date)
    report.end_date = pd.to_datetime(report.end_date)
    matrix = holdings

    current_report = pd.DataFrame(report[(report.end_date <= d)].iloc[-1]).T
    year_start = datetime(current_report.end_date[0].year, 1, 1)
    # Broadcast financial data to each client
    matrix = matrix.join(current_report, how='cross')
    for c in current_report.columns:
        if 'date' in c:
            continue
        matrix[c] = (matrix[c].replace('[\$,]', '', regex=True).astype(float)) * matrix['holdings']
    matrix['commision'] = 0
    for ir, r in matrix.iterrows():
        matrix['commision'].iloc[ir] = payment[
            (payment.investor_ID == r.investor_ID) & (payment.payment_date > year_start) & (
                    payment.payment_date < r.end_date)]['commision_discount'].sum()
    matrix['commision_pro_rata'] = -matrix['commision'].sum() * matrix['holdings']
    matrix['other_expenses_commision'] = (matrix['commision_pro_rata'] - matrix['other_expenses'] + matrix['commision'])
    intrest = calc_interest(payment, distribution, dt)

    matrix = pd.merge(matrix, intrest, on='investor_ID')
    matrix['profit_for_successes'] = matrix['rent_income'] - matrix['professional_expenses'] - matrix[
        'management_fees'] - matrix['interest_fees'] + matrix['other_expenses_commision'] + matrix[
                                         'profits_share_of_investees'] + matrix['fair_value_adjustments'] + matrix[
                                         'retained_earnings']
    success_fee = 0.2
    preferd_interest = 0.07
    lower_bound_success_fee = calc_sucsess_bound(payment, distribution, current_report.end_date[0],
                                                 intrest_fees=preferd_interest)
    upper_bound_success_fee = calc_sucsess_bound(payment, distribution, current_report.end_date[0],
                                                 intrest_fees=preferd_interest * 1 / (1 - success_fee))
    lower_bound_success_fee.rename(columns={'bound': 'lower_bound_success_fee'}, inplace=True)
    upper_bound_success_fee.rename(columns={'bound': 'upper_bound_success_fee'}, inplace=True)
    fee_bounds = pd.merge(lower_bound_success_fee, upper_bound_success_fee, on='investor_ID')
    matrix = pd.merge(matrix, fee_bounds, on='investor_ID')
    matrix['success_fee'] = ((matrix['profit_for_successes'] > matrix['lower_bound_success_fee']) & (
            matrix['profit_for_successes'] < matrix['upper_bound_success_fee']) & (
                                     matrix['profit_for_successes'] - matrix['lower_bound_success_fee'])) + (
                                    (matrix['profit_for_successes'] > matrix['upper_bound_success_fee']) & (
                                    matrix['profit_for_successes'] * success_fee))
    matrix['investment_profit_share'] = matrix['profit_for_successes'] - matrix['success_fee']

    if col_order:
        # Filter and reorder the columns based on col_order
        matrix = matrix[col_order]

    return matrix,holdings


# Function to save DataFrame to Excel and provide download link
def save_and_provide_download_link(df):
    # Save DataFrame to Excel
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
    return temp_excel_file


def main_create_final_df(year,month,day):
    col_order = ['investor_ID', 'investor_name', 'feeder', 'payment_sum', 'holdings', 'feeder_holdings',
                 'holdings_per_feeder', 'cash_and_cash_equivalents',
                 'deferred_inception_expense_net', 'related_parties', 'total_current_assets',
                 'investments_in_investees', 'real_estate_properties', 'total_long_term_assets', 'total_assets',
                 'security_deposits', 'Related Parties', 'loans', 'payables', 'total_current_liabilities', 'mortgage',
                 'total_non_current_liabilities', 'total_liabilities', 'capital_contributions',
                 'receivables_due_from_capital_contributions', 'Distributions', 'retained_earnings',
                 'accumulated_loss_from_operations', 'total_partners_capital', 'total_partners_capital_and_liabilities',
                 'rent_income', 'profits_share_of_investees', 'fair_value_adjustments', 'total_income',
                 'professional_expenses', 'interest_fees', 'management_fees', 'other_expenses', 'total_expenses',
                 'net_investment_loss', 'commision', 'commision_pro_rata', 'other_expenses_commision', 'date',
                 'interest', 'principal', 'paid_interest', 'paid_principal', 'profit_for_successes',
                 'lower_bound_success_fee', 'upper_bound_success_fee', 'success_fee',
                 'investment_profit_share']

    df,holdings = matix_per_date(datetime(year,month, day), col_order=col_order)
    coloumn_list_dollar = ['payment_sum', 'feeder_holdings',
                           'cash_and_cash_equivalents', 'deferred_inception_expense_net',
                           'related_parties', 'total_current_assets', 'investments_in_investees',
                           'real_estate_properties', 'total_long_term_assets', 'total_assets',
                           'security_deposits', 'Related Parties', 'loans', 'payables',
                           'total_current_liabilities', 'mortgage',
                           'total_non_current_liabilities', 'total_liabilities',
                           'capital_contributions', 'receivables_due_from_capital_contributions',
                           'Distributions', 'retained_earnings',
                           'accumulated_loss_from_operations', 'total_partners_capital',
                           'total_partners_capital_and_liabilities', 'rent_income',
                           'profits_share_of_investees', 'fair_value_adjustments', 'total_income',
                           'professional_expenses', 'interest_fees', 'management_fees',
                           'other_expenses', 'total_expenses', 'net_investment_loss', 'commision',
                           'commision_pro_rata', 'other_expenses_commision', 'interest',
                           'principal', 'paid_interest', 'paid_principal', 'profit_for_successes',
                           'lower_bound_success_fee',
                           'upper_bound_success_fee', 'success_fee', 'investment_profit_share']
    df[coloumn_list_dollar] = df[coloumn_list_dollar].astype(int)

    return df,holdings


if __name__ == "__main__":
    col_order = ['investor_ID', 'investor_name', 'feeder', 'payment_sum', 'holdings', 'feeder_holdings',
                 'holdings_per_feeder', 'cash_and_cash_equivalents',
                 'deferred_inception_expense_net', 'related_parties', 'total_current_assets',
                 'investments_in_investees', 'real_estate_properties', 'total_long_term_assets', 'total_assets',
                 'security_deposits', 'Related Parties', 'loans', 'payables', 'total_current_liabilities', 'mortgage',
                 'total_non_current_liabilities', 'total_liabilities', 'capital_contributions',
                 'receivables_due_from_capital_contributions', 'Distributions', 'retained_earnings',
                 'accumulated_loss_from_operations', 'total_partners_capital', 'total_partners_capital_and_liabilities',
                 'rent_income', 'profits_share_of_investees', 'fair_value_adjustments', 'total_income',
                 'professional_expenses', 'interest_fees', 'management_fees', 'other_expenses', 'total_expenses',
                 'net_investment_loss', 'commision', 'commision_pro_rata', 'other_expenses_commision', 'date',
                 'interest', 'principal', 'paid_interest', 'paid_principal', 'profit_for_successes',
                 'lower_bound_success_fee', 'upper_bound_success_fee', 'success_fee',
                 'investment_profit_share']

    x = matix_per_date(datetime(2023, 12, 31), col_order=col_order)
    save_and_provide_download_link(x)
