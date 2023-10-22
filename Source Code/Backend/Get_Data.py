import requests
import pandas as pd
import numpy as np
import time


from IPython.display import clear_output

def getData(ticker, toDate , fromDate):
    #Create Query List
    past5 = pd.DataFrame(pd.date_range(toDate, fromDate))
    query_list = []
    temp = []
    count = 0
    for row in past5[0]:
        count += 1
        temp.append(str(row.date()))
        if count == 20:
            count = 0
            query_list.append(temp)
            temp = []

    #Grab Data

    final_data = pd.DataFrame() 
    count = 0
    for query in query_list:
        count += 1
        clear_output()
        print(count)
        start = query[0]
        end = query[-1]
        url = f'https://api.tiingo.com/iex/{ticker}/prices?startDate={start}&endDate={end}&resampleFreq=1min&afterHours=false&columns=open,high,low,close,volume&token=628a6b8ff27cc43ad44f6f69e72f528e30c856b5'
        response = requests.get(url)
        print(response)
        results = pd.DataFrame(response.json())

        #Format Data

        dates = []
        times = []
        for row in range(len(results)):
            datetime = results.iloc[row]['date']
            date = datetime[:10]
            time = datetime[11:23]
            dates.append(date)
            times.append(time)
        results2 = results.drop(columns=['date'])
        results2['date'] = dates
        results2['time'] = times
        results2.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'DATE', 'TIME']
        final_data = pd.concat([final_data, results2], axis=0).reset_index(drop=True)

    fd = pd.read_csv('past2022format.csv')
    fd.set_index('DATETIME', inplace=True)

    final_data['DATETIME'] = final_data['DATE'] + ' ' + final_data['TIME']
    final_data.set_index('DATETIME', inplace=True)

    fd.update(final_data)

    fd['OPEN'] = fd['OPEN'].apply(pd.to_numeric, errors='coerce')
    fd['HIGH'] = fd['HIGH'].apply(pd.to_numeric, errors='coerce')
    fd['LOW'] = fd['LOW'].apply(pd.to_numeric, errors='coerce')
    fd['CLOSE'] = fd['CLOSE'].apply(pd.to_numeric, errors='coerce')

    fd['OPEN'] = fd['OPEN'].interpolate(method='linear')
    fd['HIGH'] = fd['HIGH'].interpolate(method='linear')
    fd['LOW'] = fd['LOW'].interpolate(method='linear')
    fd['CLOSE'] = fd['CLOSE'].interpolate(method='linear')
    fd['VOLUME'].fillna(0.0, inplace=True)
    fd['TICKER'] = ticker

    # Save the data as a csv
    fd.to_csv(ticker + '.csv')

    return fd

# Pull the data for an inputted ticker and save it as a csv
input_ticker = input("Enter a ticker: ")
data = getData(input_ticker, '2022-01-01', '2022-12-31')
data