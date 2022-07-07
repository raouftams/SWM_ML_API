import pandas as pd
from pycaret.anomaly import *
import plotly.graph_objects as go
import plotly.express as px
from data import get_data
from pattern_recognition import all_towns_trend

#df = get_data()
#
#df = df[df["code_town"] != 'S001']
#df['date'] = pd.to_datetime(df['date'])    
#df['net'] = df.groupby(by=['date', 'code_town'])['net'].transform('sum')
#df.drop_duplicates(subset=['code_town', 'date'], inplace=True)
#df2 = df.groupby(by='date').agg({
#    'net': 'sum', 
#    'pop2016': 'sum', 
#    'pop2017': 'sum', 
#    'pop2018': 'sum',
#    'pop2019': 'sum',
#    'pop2020': 'sum',
#    'pop2021': 'sum',
#    'pop2022': 'sum'
#})
#df['net'] = df.groupby('date')['net'].transform('sum')
#df.drop_duplicates('date', inplace=True)
#
#df['day'] = df.apply(lambda row: row['date'].day, axis=1)
#df['month'] = df.apply(lambda row: row['date'].month, axis=1)
#
##sort data by date
#df.sort_values(by='date', ascending=True, inplace=True)
#
##insert population column
#df['population'] = 1
#df.loc[df['date'].dt.year == 2016, 'population'] = df[df2.index.year == 2016]['pop2016']
#df.loc[df['date'].dt.year == 2017, 'population'] = df[df2.index.year == 2017]['pop2017']
#df.loc[df['date'].dt.year == 2018, 'population'] = df[df2.index.year == 2018]['pop2018']
#df.loc[df['date'].dt.year == 2019, 'population'] = df[df2.index.year == 2019]['pop2019']
#df.loc[df['date'].dt.year == 2020, 'population'] = df[df2.index.year == 2020]['pop2020']
#df.loc[df['date'].dt.year == 2021, 'population'] = df[df2.index.year == 2021]['pop2021']
#df.loc[df['date'].dt.year == 2022, 'population'] = df[df2.index.year == 2022]['pop2022']
#
#
##change columns order and put the net column as class column
#df["value"] = df['net']/1000
#df['value'] = df['value'].astype(int)
#
#df['value'] = df['value'].rolling(3).mean()
#df.dropna(axis=0, inplace=True)
##group data
#df['value'] = df.groupby('date')['value'].transform('sum')
#df = df.drop_duplicates('date')
#df = df.set_index('date')
#df['date'] = df.index
#df = df.fillna(df.median())    
#df.drop(['latitude', 'longitude', 'temperature', 'day', 'vent', 'season', 'date', 'code_ticket', 'code_town', 'code_unity', 'net', 'date_hijri', 'cet', 'pop2016', 'pop2017', 'pop2018', 'pop2019', 'pop2020', 'pop2021','pop2022'], axis=1, inplace=True)
#corr_matrix = df.corr()
#print(df['season'])
#nominal.associations(df, mark_columns=True)
#plt.show()

df = all_towns_trend(1)
df['values'] = df['values'].rolling(3).mean()

s = setup(df, session_id = 123)
 
iforest = create_model('iforest', fraction = 0.02)
iforest_results = assign_model(iforest)
iforest_results.head()
iforest_results[iforest_results['Anomaly'] == 1].head()
# plot value on y-axis and date on x-axis
fig = px.line(iforest_results, x=iforest_results.index, y="values", title='WASTE QUANTITY - UNSUPERVISED ANOMALY DETECTION')
# create list of outlier_dates
outlier_dates = iforest_results[iforest_results['Anomaly'] == 1].index
# obtain y value of anomalies to plot
y_values = [iforest_results.loc[i]['values'] for i in outlier_dates]
fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))
        
fig.show()