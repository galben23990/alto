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


service_account_info = st.secrets["FIREBASE_SERVICE_ACCOUNT"]
# Load the service account credentials from Streamlit secrets

# Convert AttrDict to a regular dictionary
service_account_dict = dict(service_account_info)

# Write the credentials to a temporary file
temp_cred_file = "temp_service_account.json"
with open(temp_cred_file, 'w') as file:
    json.dump(service_account_dict, file)

# Authorize pygsheets with the service account
gc = pygsheets.authorize(service_file=temp_cred_file)


def get_as_df_float(worksheet):
    df = worksheet.get_as_df()
    for col in df.columns:
        try:
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
        except:
            try:
                df[col] = df[col].astype(float)
            except:
                pass
    return df
def get_data():
    sh = gc.open("Alto")
    wks = sh.worksheet_by_title("payment")
    payment = get_as_df_float(wks)
    payment.payment_date=pd.to_datetime(payment.payment_date)
    wks = sh.worksheet_by_title("commitment")
    commitment = get_as_df_float(wks)
    commitment.commitment_date = pd.to_datetime(commitment.commitment_date)
    wks = sh.worksheet_by_title("report")
    report = get_as_df_float(wks)
    wks = sh.worksheet_by_title("distribution")
    distribution = get_as_df_float(wks)
    ## conert to numbers
    # distribution['distribution']=distribution['distribution'].replace('[\$,]', '', regex=True).astype(float)
    return payment,commitment,report,distribution

def holding_per_date(payment,d):
    dt = datetime.combine(d, datetime.min.time())
    payment = payment[payment.payment_date <= dt]
    #convert these columns to numbers ['payment_sum', 'calling_commitment', 'debt_from_commitment'] if empty set to 0
    payment['payment_sum'].fillna(0, inplace=True)
    payment['calling_commitment'].fillna(0, inplace=True)
    payment['debt_from_commitment'].fillna(0, inplace=True)
    payment['payment_sum'] = payment['payment_sum'].replace('[\$,]', '', regex=True).astype(float)
    payment['calling_commitment'] = payment['calling_commitment'].replace('[\$,]', '', regex=True).astype(float)
    payment['debt_from_commitment'] = payment['debt_from_commitment'].replace('[\$,]', '', regex=True).astype(float)


    ## calculate the holdings per investor per date sum over payment_sum calling_commitment and debt_from_commitment
    holdings = (payment.groupby(by=['investor_ID', 'investor_name', 'feeder'])[['payment_sum', 'calling_commitment', 'debt_from_commitment']].sum())
    holdings = holdings.reset_index()
    holdings["holdings"] = holdings['payment_sum'] / holdings['payment_sum'].sum()

    holdings_feeder = (payment.groupby(by=['feeder'])['payment_sum'].sum())
    holdings_feeder = holdings_feeder.reset_index()
    holdings_feeder.rename(columns={'payment_sum': 'feeder_holdings'}, inplace=True)
    holdings=pd.merge(holdings, holdings_feeder, on='feeder')
    holdings["holdings_per_feeder"] = holdings['payment_sum'] / holdings['feeder_holdings']
    return holdings

def calc_interest(payment, distribution,cur_date,investment_part=0.7,intrest_fees=0.1):
    final_interest=[]
    for name in payment.investor_name.unique():
        CF_df=payment[payment.investor_name==name]
        if any(CF_df['feeder']=="Alto real estate Holdings US, LP"):
            continue
        CF_df=CF_df[['payment_date','payment_sum']]
        CF_df.rename(columns={'payment_date':'date','payment_sum':'cash_flow'},inplace=True)
        distribution_id=distribution[distribution.investor_name==name]
        distribution_id=distribution_id[['date','distribution']]
        distribution_id['distribution']=-distribution_id['distribution']
        distribution_id.rename(columns={'distribution':'cash_flow'},inplace=True)
        distribution_id['date']=pd.to_datetime(distribution_id['date'])
        CF_df=pd.concat([CF_df,distribution_id])
        CF_df=CF_df[CF_df.date<=cur_date]
        CF_df.reset_index(inplace=True,drop=True)
        CF_df["is_quarter"]=False
        CF_df[CF_df["cash_flow"]<0]["is_quarter"]=True
        for quarter in ['03-31','06-30','09-30','12-31']:
            tmp_date=datetime(cur_date.year,int(quarter.split('-')[0]),int(quarter.split('-')[1]))
            if tmp_date<=cur_date:
                CF_df.loc[len(CF_df.index)]=[tmp_date,0.0,True]
        CF_df.loc[len(CF_df.index)]=[cur_date,0.0,True]
        CF_df.sort_values(by='date',inplace=True)
        CF_df.reset_index(inplace=True,drop=True)
        CF_df['principal']=0
        CF_df['interest']=0
        CF_df['paid_interest']=0
        CF_df['paid_principal']=0
        for i,r in CF_df.iterrows():
            if i == 0:
                assert (r['cash_flow'] >= 0)
                CF_df.loc[i, 'principal'] = r['cash_flow'] * investment_part
                CF_df.loc[i, 'interest'] = 0
                CF_df.loc[i, 'paid_interest'] = 0
                CF_df.loc[i, 'paid_principal'] = 0
            else:
                days = (r['date'] - CF_df.loc[i - 1, 'date']).days
                if CF_df.loc[i,'is_quarter']:
                    CF_df.loc[i, 'interest'] = CF_df.loc[i - 1, 'interest'] + (CF_df.loc[i - 1, 'principal'] + CF_df.loc[i - 1, 'interest']) * (intrest_fees * (days / 365))
                else:
                    CF_df.loc[i, 'interest'] = CF_df.loc[i - 1, 'interest'] + (CF_df.loc[i - 1, 'principal'] + CF_df.loc[i - 1, 'interest']) * (intrest_fees * (days / 365))
                CF_df.loc[i, 'paid_interest'] = CF_df.loc[i-1,'paid_interest']
                CF_df.loc[i, 'paid_principal'] = CF_df.loc[i - 1, 'paid_principal']
                if r.cash_flow>0:
                    CF_df.loc[i,'principal']=CF_df.loc[i-1,'principal']+r['cash_flow']*investment_part
                else:
                    if -r['cash_flow']<CF_df.loc[i,'interest']:
                        CF_df.loc[i, 'interest']=CF_df.loc[i,'interest']+r['cash_flow']
                        CF_df.loc[i, 'principal']=CF_df.loc[i-1,'principal']
                        CF_df.loc[i, 'paid_interest']=CF_df.loc[i,'paid_interest']-r['cash_flow']
                    else:
                        CF_df.loc[i, 'paid_interest']=CF_df.loc[i, 'paid_interest']+CF_df.loc[i,'interest']
                        CF_df.loc[i, 'paid_principal']=CF_df.loc[i, 'paid_principal']-(r['cash_flow']+CF_df.loc[i,'interest'])
                        CF_df.loc[i, 'principal']=CF_df.loc[i-1,'principal']+(r['cash_flow']+CF_df.loc[i,'interest'])
                        CF_df.loc[i, 'interest'] = 0
        interest_id=pd.DataFrame({'investor_name':name,'date':CF_df['date'].iloc[-1],'interest':CF_df['interest'].iloc[-1],'principal':CF_df['principal'].iloc[-1],'paid_interest':CF_df['paid_interest'].iloc[-1],'paid_principal':CF_df['paid_principal'].iloc[-1]},index=[0])
        final_interest.append(interest_id)
    return pd.concat(final_interest)

def calc_sucsess_bound(payment, distribution,cur_date,investment_part=1,intrest_fees=0.07):
    final_interest=[]
    for name in payment.investor_name.unique():
        CF_df=payment[payment.investor_name==name]
        CF_df=CF_df[['payment_date','payment_sum']]
        CF_df.rename(columns={'payment_date':'date','payment_sum':'cash_flow'},inplace=True)
        distribution_id=distribution[distribution.investor_name==name]
        distribution_id=distribution_id[['date','distribution']]
        distribution_id['distribution']=-distribution_id['distribution']
        distribution_id.rename(columns={'distribution':'cash_flow'},inplace=True)
        distribution_id['date']=pd.to_datetime(distribution_id['date'])
        CF_df=pd.concat([CF_df,distribution_id])
        CF_df=CF_df[CF_df.date<=cur_date]
        CF_df.sort_values(by='date',inplace=True)
        CF_df.reset_index(inplace=True,drop=True)
        CF_df['lower_bound']=0

        days = (cur_date - CF_df['date'])
        # convert days to float number of days
        days = days.dt.total_seconds() / (24 * 60 * 60)
        bound=(CF_df['cash_flow']*investment_part*(intrest_fees*(days/365))).sum()
        bound_id=pd.DataFrame({'investor_name':name,'date':cur_date,'bound':bound},index=[0])
        final_interest.append(bound_id)
    return pd.concat(final_interest)

def matix_per_date(d):
    payment, commitment, report, distribution=get_data()
    # distribution=disribution_make_table(distribution,payment)
    dt = datetime.combine(d, datetime.min.time())
    holdings = holding_per_date(payment,d)
    openning_holdings = holding_per_date(payment,datetime(d.year,1,1))
    ## join the holdings with the openning holdings
    holdings=pd.merge(holdings,openning_holdings[['investor_ID','investor_name','feeder','debt_from_commitment']],on=['investor_ID','investor_name','feeder'],suffixes=('','_openning'),how='outer')
    holdings=holdings.fillna(0)
    holdings['debt_from_commitment_movement']=holdings['debt_from_commitment']-holdings['debt_from_commitment_openning']
    report=report.set_index("name").transpose()
    report.start_date=pd.to_datetime(report.start_date)
    report.end_date=pd.to_datetime(report.end_date)
    matrix=holdings
    ## add comitment sum to the matrix
    matrix=pd.merge(matrix,commitment[['investor_ID','commitment_sum']],on='investor_ID')
    # if no report is prior to the date then take an empty report with zero values
    if len(report[(report.end_date<=d)])==0:
        current_report=pd.DataFrame(report.iloc[0]).T
        #change the date to the current date and the values to 0
        current_report['end_date']=d
        current_report['start_date']=d
        for c in current_report.columns:
            if 'date' in c:
                continue
            current_report[c]=0
    else:
        current_report=pd.DataFrame(report[(report.end_date<=d)].iloc[-1]).T
    year_start = datetime(current_report.end_date[0].year, 1, 1)
    # Broadcast financial data to each client
    matrix = matrix.join(current_report, how='cross')
    for c in current_report.columns:
        if 'date' in c:
            continue
        matrix[c]=(matrix[c].replace('[\$,]', '', regex=True).astype(float))*matrix['holdings']
    matrix['commision']=0
    distribution['date']=pd.to_datetime(distribution['date'])
    for ir,r in matrix.iterrows():
        matrix['Distributions'].iloc[ir]=distribution[(distribution.investor_name==r.investor_name)*(distribution.date>r.start_date)*(distribution.date<r.end_date)]['distribution'].sum()
        matrix['commision'].iloc[ir]=payment[(payment.investor_ID==r.investor_ID)*(payment.payment_date>year_start)*(payment.payment_date<r.end_date)]['commision_discount'].sum()
    matrix['commision_pro_rata']=-matrix['commision'].sum()*matrix['holdings']
    matrix['other_expenses_commision']=(matrix['commision_pro_rata']-matrix['other_expenses']+matrix['commision'])
    intrest=calc_interest(payment,distribution,dt)
    openintrest = calc_interest(payment, distribution, datetime.combine(datetime(d.year,1,1), datetime.min.time()))
    ## merge the intrest with the openintrest to get the movement
    intrest=pd.merge(intrest,openintrest[['investor_name','interest','principal','paid_interest','paid_principal']],on='investor_name',suffixes=('','_openning'),how='outer')
    intrest=intrest.fillna(0)
    intrest['interest_movement']=intrest['interest']-intrest['interest_openning']
    intrest['principal_movement']=intrest['principal']-intrest['principal_openning']
    intrest['paid_interest_movement']=intrest['paid_interest']-intrest['paid_interest_openning']
    intrest['paid_principal_movement']=intrest['paid_principal']-intrest['paid_principal_openning']
    intrest['interest_fees_investor']=intrest['interest_movement']+intrest['paid_interest_movement']


    # matrix[]


    matrix=pd.merge(matrix,intrest,on='investor_name',how='left')
    # fill empty values with 0 of only columns from intrest
    columns=['interest','principal','paid_interest','paid_principal','interest_movement','principal_movement','paid_interest_movement','paid_principal_movement','interest_fees_investor']
    matrix[columns]=matrix[columns].fillna(0)

    matrix['profit_for_successes']=(matrix['rent_income']-matrix['professional_expenses']-matrix['management_fees']-matrix['interest_fees']+matrix['other_expenses_commision']+matrix['profits_share_of_investees']+matrix['fair_value_adjustments']+matrix['retained_earnings'])

    success_fee=0.2
    preferd_interest=0.07
    lower_bound_success_fee=calc_sucsess_bound(payment, distribution, current_report.end_date[0],intrest_fees=preferd_interest)
    upper_bound_success_fee = calc_sucsess_bound(payment, distribution, current_report.end_date[0],intrest_fees=preferd_interest*1/(1-success_fee))
    lower_bound_success_fee.rename(columns={'bound':'lower_bound_success_fee'},inplace=True)
    upper_bound_success_fee.rename(columns={'bound':'upper_bound_success_fee'},inplace=True)
    fee_bounds=pd.merge(lower_bound_success_fee,upper_bound_success_fee,on='investor_name')
    matrix=pd.merge(matrix,fee_bounds,on='investor_name')
    matrix['success_fee']=((matrix['profit_for_successes']>matrix['lower_bound_success_fee'])\
                          *(matrix['profit_for_successes']<matrix['upper_bound_success_fee'])\
                          *(matrix['profit_for_successes']-matrix['lower_bound_success_fee']))\
                          +((matrix['profit_for_successes']>matrix['upper_bound_success_fee'])\
                                                  *(matrix['profit_for_successes']*success_fee))

    matrix['investment_profit_share']=matrix['profit_for_successes']-matrix['success_fee']
    matrix.fillna(0.0, inplace=True)
    coloumn_list_dollar=['payment_sum','feeder_holdings','debt_from_commitment','calling_commitment',
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
     'commision_pro_rata', 'other_expenses_commision','interest',
     'principal', 'paid_interest', 'paid_principal', 'profit_for_successes',
     'lower_bound_success_fee',
     'upper_bound_success_fee', 'success_fee', 'investment_profit_share',
     'interest_movement', 'principal_movement', 'paid_interest_movement', 'paid_principal_movement', 'interest_fees_investor',
     'interest_openning', 'principal_openning', 'paid_interest_openning', 'paid_principal_openning'
     ]
    matrix[coloumn_list_dollar]=matrix[coloumn_list_dollar].round().astype(int)
    matrix['other_expenses_commision']=-matrix['other_expenses_commision']
    matrix['total_expenses']+=matrix['other_expenses_commision']-matrix['other_expenses']
    matrix['net_investment_loss']=matrix['total_income']-matrix['total_expenses']+matrix['retained_earnings']

    matrix.drop(columns=['cash_and_cash_equivalents', 'deferred_inception_expense_net',
     'related_parties', 'total_current_assets', 'investments_in_investees',
     'real_estate_properties', 'total_long_term_assets', 'total_assets',
     'security_deposits', 'Related Parties', 'loans', 'payables',
     'total_current_liabilities', 'mortgage',
     'total_non_current_liabilities', 'total_liabilities','total_partners_capital_and_liabilities','receivables_due_from_capital_contributions',],inplace=True)
    return matrix

def matix_per_date_including_openning_movement(d):
    open_matrix=matix_per_date(datetime(d.year,1,1))
    close_matrix=matix_per_date(d)
    final_matrix=close_matrix
    final_matrix['balance']=0
    final_matrix['opening_balance']=0
    ## for each investor calculate the movement from the openning matrix if the investor is not in the openning matrix then the movement is the value of the close matrix
    coloumns_for_open_close=['payment_sum','Distributions','success_fee','net_investment_loss']
    for investor in close_matrix.investor_name.unique():
        open_matrix_investor=open_matrix[open_matrix.investor_name==investor]
        iloc=close_matrix[close_matrix.investor_name==investor].index
        if len(open_matrix_investor)==0:
            for c in coloumns_for_open_close:
                final_matrix.loc[iloc,c+'_movement']=final_matrix.loc[iloc,c]
                final_matrix.loc[iloc,c+'_open']=0
        else:
            for c in coloumns_for_open_close:
                final_matrix.loc[iloc,c+'_movement']=final_matrix.loc[iloc,c]-open_matrix_investor[c]
                final_matrix.loc[iloc,c+'_open']=open_matrix_investor[c]
        final_matrix.loc[iloc,'balance']=final_matrix.loc[iloc,'payment_sum']+final_matrix.loc[iloc,'Distributions']-final_matrix.loc[iloc,'success_fee']+final_matrix.loc[iloc,'net_investment_loss']
        final_matrix.loc[iloc,'opening_balance']=final_matrix.loc[iloc,'payment_sum_open']+final_matrix.loc[iloc,'Distributions_open']-final_matrix.loc[iloc,'success_fee_open']+final_matrix.loc[iloc,'net_investment_loss_open']
    return final_matrix



if __name__ == "__main__":
    # matix_per_date(datetime(2023,12,31))
    matix_per_date_including_openning_movement(datetime(2023,12,31))
    # all_creds = []
    # for i, row in cred_df.iterrows():
    #     # if (row.latest_date_uploaded=='') or (datetime.datetime.fromisoformat(row.latest_date_uploaded)<datetime.datetime.now()-datetime.timedelta(5)):
    #     tmp_dict = row.to_dict()
    #     all_creds.append(tmp_dict)
    # dict_cf = {}
