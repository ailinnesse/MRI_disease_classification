import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import requests
import asyncio
from pyppeteer import launch

# Function to download CSV files from GitHub
def download_csv_from_github(url, file_name):
    response = requests.get(url)
    with open(file_name, 'wb') as file:
        file.write(response.content)

# URLs of the raw CSV files in your GitHub repository
gc_url = 'https://raw.githubusercontent.com/ailinnesse/MRI_disease_classification/main/code/gc_forecast.csv'
gc_summary_url = 'https://raw.githubusercontent.com/ailinnesse/MRI_disease_classification/main/code/gc_summary.csv'
pay_periods_url = 'https://raw.githubusercontent.com/ailinnesse/MRI_disease_classification/main/code/pay_periods.csv'

# Download the CSV files
download_csv_from_github(gc_url, 'gc_forecast.csv')
download_csv_from_github(gc_summary_url, 'gc_summary.csv')
download_csv_from_github(pay_periods_url, 'pay_periods.csv')

# Load data with error handling for bad lines
gc = pd.read_csv('gc_forecast.csv', on_bad_lines='skip')
gc_summary = pd.read_csv('gc_summary.csv', on_bad_lines='skip')
pay_periods = pd.read_csv('pay_periods.csv', on_bad_lines='skip')

# Change StartDate and EndDate column to DateTime to be able to compare to the Months for calculations
gc['StartDate'] = pd.to_datetime(gc['StartDate'])
gc['EndDate'] = pd.to_datetime(gc['EndDate'])
pay_periods['Month'] = pd.to_datetime(pay_periods['Month'])

# Add Hours to the pay_period dataframe
pay_periods['Hours'] = pay_periods['Pay Periods'].apply(lambda x: 160 if x == 2 else 240)

def hours_in_month(pay_periods, last_day_of_month):
    hours = pay_periods.loc[pay_periods['Month'] == last_day_of_month, 'Hours'].values
    return hours[0]

def first_day_of_month(date):
    return date.replace(day=1)

def last_day_of_month(date):
    if date.month == 12:
        next_month = datetime(date.year + 1, 1, 1)
    else:
        next_month = datetime(date.year, date.month + 1, 1)
    return next_month - timedelta(days=1)

def hours_for_work_starting_in_month(hours_in_month, first_day_of_month, start_date):
    workdays = np.busday_count(first_day_of_month.date(), start_date.date())
    return hours_in_month - (workdays * 8)

def hours_for_work_ending_in_month(hours_in_month, end_date, last_day_of_month):
    workdays = np.busday_count(end_date.date(), last_day_of_month.date())
    return hours_in_month - (workdays * 8)

def monthly_planned_hours(start_date, end_date, first_day_of_month, last_day_of_month, hours_for_work_starting_in_month, hours_for_work_ending_in_month, hours_in_month):
    if start_date > last_day_of_month:
        return 0
    elif end_date < first_day_of_month:
        return 0
    elif end_date > last_day_of_month:
        if start_date > first_day_of_month:
            return hours_for_work_starting_in_month
        else:
            return hours_in_month
    else:
        return hours_for_work_ending_in_month

current_date = datetime.today()
max_end_date = gc['EndDate'].max()
months = pd.date_range(start=current_date.replace(day=1) + pd.DateOffset(months=1), end=max_end_date, freq='MS')

for month in months:
    first_day = first_day_of_month(month)
    last_day = last_day_of_month(month)
    gc[f'PlannedHours_{month.strftime("%Y_%m")}'] = gc.apply(
        lambda row: monthly_planned_hours(
            row['StartDate'], row['EndDate'], first_day, last_day,
            hours_for_work_starting_in_month(hours_in_month(pay_periods, last_day), first_day_of_month(row['StartDate']), row['StartDate']),
            hours_for_work_ending_in_month(hours_in_month(pay_periods, last_day), row['EndDate'], last_day),
            hours_in_month(pay_periods, last_day)
        ), axis=1
    )

async def html_to_pdf(html_content, output_path):
    browser = await launch()
    page = await browser.newPage()
    await page.setContent(html_content)
    await page.pdf({'path': output_path, 'format': 'A4'})
    await browser.close()

# Streamlit app
st.title("GC Forecast App")

project_display_name = st.text_input("Project Display Name")

download_option = st.selectbox("Select download option", ["CSV", "PDF"])

if st.button("Generate and Download"):
    if download_option == "CSV":
        csv_file = f"{project_display_name}_output.csv"
        gc.to_csv(csv_file, index=False)
        st.success(f"CSV file {csv_file} generated successfully!")
        with open(csv_file, 'rb') as file:
            st.download_button(label="Download CSV", data=file.read(), file_name=csv_file, mime='text/csv')
    elif download_option == "PDF":
        html_content_gc = gc.to_html(classes='table table-striped table-bordered', index=False)
        html_content_gc_summary = gc_summary.to_html(classes='table table-striped table-bordered', index=False)
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>GC Forecast</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                h1 {{
                    text-align: left;
                    font-size: 24px;
                    margin-bottom: 20px;
                }}
                h2 {{
                    text-align: left;
                    font-size: 20px;
                    margin-top: 20px;
                }}
                .table {{
                    max-width: 100%;
: top;
                    border-top: 1px solid #dee2e6;
                }}
                .table thead th {{
                    vertical-align: bottom;
                    border-bottom: 2px solid #dee2e6;
                    font-weight: bold;
                }}
                .table-bordered {{
                    border: 1px solid #dee2e6;
                }}
                .table-bordered th, .table-bordered td {{
                    border: 1px solid #dee2e6;
                }}
                .table-striped tbody tr:nth-of-type(odd) {{
                    background-color: rgba(0, 0, 0, 0.05);
                }}
                .summary-table {{
                    margin-left: 0;
                }}
            </style>
        </head>
        <body>
            <h1>GC Forecast</h1>
            <h2>Summary</h2>
            <div class="summary-table">
                {html_content_gc_summary}
            </div>
            <h2>Details</h2>
            {html_content_gc}
        </body>
        </html>
        """
        pdf_file = f"{project_display_name}_gc_forecast.pdf"
        asyncio.get_event_loop().run_until_complete(html_to_pdf(html_content, pdf_file))
        st.success(f"PDF file {pdf_file} generated successfully!")
        with open(pdf_file, 'rb') as file:
            st.download_button(label="Download PDF", data=file.read(), file_name=pdf_file, mime='application/pdf')
