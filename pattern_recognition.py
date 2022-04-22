import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from tslearn.clustering import TimeSeriesKMeans

#local imports
from data import get_data, get_rotations_by_hour, get_rotations_number
from utilities import savePkl, openPkl

#apriori algorithme
def apriori_freq_patterns(df, min_support):
    #encode transactions
    te = TransactionEncoder()
    te_array = te.fit(df.to_numpy()).transform(df.to_numpy())
    #transform to pandas dataframe
    te_df = pd.DataFrame(te_array, columns=te.columns_)
    #apply algorithme
    frq_items = apriori(te_df, min_support = min_support, use_colnames = True)
    
    return frq_items

#extract association rules
def get_association_rules(freq_items, metric="lift", min_threshold=1):
    rules = association_rules(freq_items, metric =metric, min_threshold = min_threshold)
    rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])

    return rules

#get seasonality of data
def data_seasonality(df, column, model="multiplicative", period=365):
    """
    Args:
        df: pandas dataframe with 2 columns (date and any values)
        column: values columns name
        model: "multiplicative" or "additive" data decomposition
        period: the period of the series (in daily data it represents the number of days)
    returns:
        pandas series  
    """
    #transform date to datetime and set it as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    #create a copy for the analysis
    analysis = df[[column]].copy()
    decompose_result_mult = seasonal_decompose(analysis, model=model, period=period, extrapolate_trend='freq')
    seasonal = decompose_result_mult.seasonal

    return seasonal

#get the trend of data
def data_trend(df, column, model="multiplicative", period=365):
    """
    Args:
        df: pandas dataframe with 2 columns (date and any values)
        column: values columns name
        model: "multiplicative" or "additive" data decomposition
        period: the period of the series (in daily data it represents the number of days)
    returns:
        pandas series  
    """
    #transform date to datetime and set it as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    #create a copy for the analysis
    analysis = df[[column]].copy()
    decompose_result_mult = seasonal_decompose(analysis, model=model, period=period, extrapolate_trend='freq')
    trend = decompose_result_mult.trend

    return trend

#clustering of seasons and waste quantities
def season_qte_cluster(df):
    #prepare data
    #discretize qte data
    #df["net"] = pd.qcut(df["net"], q=5)
    #encode data
    le = LabelEncoder()
    cls_df = df.copy()
    print(cls_df)
    #for col in cls_df.columns:
    #    cls_df[col] = le.fit_transform(cls_df[col])
    #cls_df["values"] = le.fit_transform(cls_df["values"])


    #split data
    X = StandardScaler().fit_transform(cls_df)

    cls = KMeans(n_clusters=4)
    cls.fit_transform(X)
    print(cls.score(X))
    y_kmeans = cls.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = cls.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()


#holiday qte using
def holiday_qte():
    df = get_data()
    #group by date 
    df["net"] = df.groupby(by='date')["net"].transform('sum')
    df.drop_duplicates(subset='date', inplace=True)
    holidays = df[["net", "holiday"]]
    holidays["net"] = holidays.groupby(by="holiday")["net"].transform("mean")
    holidays.drop_duplicates(subset=['holiday', 'net'], inplace=True)

    return holidays

#holiday qte by town
def holiday_qte_town(code_town):
    df = get_data()
    holidays = df[["net", "holiday", "code_town"]]
    holidays = holidays.loc[holidays["code_town"] == code_town]
    holidays["net"] = holidays.groupby(by="holiday")["net"].transform("mean")
    holidays.drop_duplicates(subset=['holiday', 'net'], inplace=True)

    return holidays

#holiday qte by unity
def holiday_qte_unity(code_unity):
    df = get_data()
    holidays = df[["net", "holiday", "code_unity"]]
    holidays = holidays.loc[holidays["code_unity"] == code_unity]
    holidays["net"] = holidays.groupby(by="holiday")["net"].transform("mean")
    holidays.drop_duplicates(subset=['holiday', 'net'], inplace=True)

    return holidays

#season qte by town
def season_qte_town(code_town):
    df = get_data()
    seasons_df = df[["net", "season", "code_town"]]
    seasons_df = seasons_df.loc[seasons_df["code_town"] == code_town]
    seasons_df["net"] = seasons_df.groupby(by="season")["net"].transform("mean")
    seasons_df.drop_duplicates(subset=['season', 'net'], inplace=True)

    return seasons_df

#season qte by unity
def season_qte_unity(code_unity):
    df = get_data()
    seasons_df = df[["net", "season", "code_unity"]]
    seasons_df = seasons_df.loc[seasons_df["code_unity"] == code_unity]
    seasons_df["net"] = seasons_df.groupby(by="season")["net"].transform("mean")
    seasons_df.drop_duplicates(subset=['season', 'net'], inplace=True)

    return seasons_df

#season qte all towns
def season_qte():
    df = get_data()
    #goup by date
    df["net"] = df.groupby(by='date')["net"].transform('sum')
    df.drop_duplicates(subset='date', inplace=True)
    seasons_df = df[["net", "season"]]
    seasons_df["net"] = seasons_df.groupby(by='season')["net"].transform('mean')
    seasons_df.drop_duplicates(subset='season', inplace=True)
    return seasons_df


#monthly waste qte 
def qte_change_rate_by_season():
    df = get_data()
    #get towns values
    towns = np.unique(df["code_town"].to_numpy())
    #dict town: dataframe
    towns_dict = {}
    for town in towns:
        if town != 'S001':
            towns_dict[town] = season_qte_town(town)
            
    
    town_change_rate = {}
    for town in towns_dict.keys():
        town_df = towns_dict[town]
        winter = town_df.loc[town_df['season'] == 'winter']["net"].to_numpy()[0]
        spring = town_df.loc[town_df['season'] == 'spring']["net"].to_numpy()[0]
        summer = town_df.loc[town_df['season'] == 'summer']["net"].to_numpy()[0]
        autumn = town_df.loc[town_df['season'] == 'autumn']["net"].to_numpy()[0]
        town_change_rate[town] = {
            "summer": 100 - (((winter+spring+autumn)/3)*100)/summer,
            "winter": 100 - (((summer+spring+autumn)/3)*100)/winter,
            "spring": 100 - (((winter+summer+autumn)/3)*100)/spring,
            "autumn": 100 - (((winter+spring+summer)/3)*100)/autumn
        }

    return town_change_rate

def holiday_qte_svc(df):
    #prepare data
    #group data by town and date
    df["net"] = df.groupby(['code_town', 'date'])["net"].transform('sum')
    df.drop_duplicates(subset=['code_town', 'date'], inplace=True)

    #group data by date
    df["net"] = df.groupby(by='date')["net"].transform('mean')
    df.drop_duplicates('date', inplace=True)
    #remove useless columns
    df = df.drop(['date', 'code_town', "hijri_month", 'date_hijri', "cet", 'code_ticket', 'latitude', 'longitude', 'temperature'], axis=1)
    print(df)

    #encode categorical data
    le = LabelEncoder()
    df["holiday"] = le.fit_transform(df["holiday"])

    #normalize data
    df["year"] = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min())
    df["holiday"] = (df["holiday"] - df["holiday"].min()) / (df["holiday"].max() - df["holiday"].min())
    df["net"] = df["net"].astype(int)
    years = np.unique(df["year"].to_numpy())
    for year in years:
        df.loc[df["year"] == year]["net"] = (df.loc[df["year"] == year]["net"] - df.loc[df["year"] == year]["net"].min()) / (df.loc[df["year"] == year]["net"].max() - df.loc[df["year"] == year]["net"].min())
    #zscore normalization
    #df["net"] = (df["net"] - df["net"].mean()) / df["net"].std()
    
    #discretize data
    df["net"] = pd.qcut(df["net"], q=2)
    #encode data
    df["net"] = le.fit_transform(df["net"])

    #split data
    y = df.pop("net")
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = SVC(C=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))


def plot_data(df):
    #group data by town and date
    df["net"] = df.groupby(['code_town', 'date'])["net"].transform('sum')
    df.drop_duplicates(subset=['code_town', 'date'], inplace=True)

    #group data by date
    df["net"] = df.groupby(by='date')["net"].transform('mean')
    df.drop_duplicates('date', inplace=True)


    #plot mean quantity by day
    #ax = df.plot(x='date', y='net', figsize=(12,6))
    #plt.show()

    #seasonality
    seasonal = data_seasonality(df.copy(), "net", period=365)
    holiday_qet(df)
    trend = data_trend(df.copy(), "net")
    trend.plot()
    plt.show()
    

    holidays = df[["date", "temperature"]]
    holidays["date"] = pd.to_datetime(holidays["date"])
    #holidays["temperature"] = pd.qcut(holidays["temperature"], q=4)
    #holidays["temperature"] = holidays["temperature"].astype(str)
    seasonal_df = seasonal.to_frame("values")
    seasonal_df = seasonal_df.reset_index()

    seasonal_df = seasonal_df.merge(holidays, on='date', how='inner')
    #seasonal_df["values"] = pd.qcut(seasonal_df["values"], q=3)
    #seasonal_df["values"] = seasonal_df["values"].astype(str)
    seasonal_df = seasonal_df.drop("date", axis=1)

    #frq_items = apriori_freq_patterns(seasonal_df, 0.13)
    #print(frq_items)
    #rules = get_association_rules(frq_items)
    #print(rules)

"""---------- waste quantity --------------"""
def trend_by_town(period: int):
    df = get_data()
    #group data by town and date
    df["net"] = df.groupby(['code_town', 'date'])["net"].transform('sum')
    df.drop_duplicates(subset=['code_town', 'date'], inplace=True)

    towns = np.unique(df['code_town'].to_numpy())
    
    town_trend = {}
    for town in towns:
        if town != "S001":
            town_df = df.loc[df["code_town"] == town]
            town_df = town_df.set_index('date')
            town_df.index = pd.to_datetime(town_df.index)
            town_df = town_df.reindex(pd.date_range(start="2016-01-01", end="2021-12-31"), fill_value=town_df["net"].mean())
            town_df["date"] = town_df.index
            trend_df = data_trend(town_df.copy(), "net", model="multiplicative", period=period).to_frame("values")
            town_trend[town] = trend_df
    
    return town_trend

def trend_by_town_year(town_code, period: int, year: int):
    df = get_data()
    #group data by town and date
    df["net"] = df.groupby(['code_town', 'date'])["net"].transform('sum')
    df.drop_duplicates(subset=['code_town', 'date'], inplace=True)

    town_df = df[df["code_town"] == town_code]
    town_df["date"] = pd.to_datetime(town_df["date"])
    print(town_df)
    town_df = town_df[town_df['date'].dt.year == year]
    town_df = town_df.set_index('date')
    town_df["date"] = town_df.index
    
    trend_df = data_trend(town_df.copy(), "net", model="additive", period=period).to_frame("values")
    
    return trend_df

def trend_by_unity(period: int):
    df = get_data()
    #group data by town and date
    df["net"] = df.groupby(['code_unity', 'date'])["net"].transform('sum')
    df.drop_duplicates(subset=['code_unity', 'date'], inplace=True)

    units = np.unique(df['code_unity'].to_numpy())
    
    unity_trend = {}
    for unity in units:
        if unity != "U00":
            unity_df = df.loc[df["code_unity"] == unity]
            unity_df = unity_df.set_index('date')
            unity_df.index = pd.to_datetime(unity_df.index)
            unity_df = unity_df.reindex(pd.date_range(start="2016-01-01", end="2021-12-31"), fill_value=unity_df["net"].mean())
            unity_df["date"] = unity_df.index
            trend_df = data_trend(unity_df.copy(), "net", model="additive", period=period).to_frame("values")
            unity_trend[unity] = trend_df
    
    return unity_trend

def trend_by_unity_year(unity_code, period: int, year: int):
    df = get_data()
    #group data by town and date
    df["net"] = df.groupby(['code_unity', 'date'])["net"].transform('sum')
    df.drop_duplicates(subset=['code_unity', 'date'], inplace=True)

    unity_df = df[df["code_unity"] == unity_code]
    unity_df["date"] = pd.to_datetime(unity_df["date"])
    unity_df = unity_df[unity_df['date'].dt.year == year]
    unity_df = unity_df.set_index('date')
    unity_df["date"] = unity_df.index
    
    trend_df = data_trend(unity_df.copy(), "net", model="additive", period=period).to_frame("values")
    return trend_df

def all_towns_trend(period: int):
    df = get_data()
    #group data by date
    df["net"] = df.groupby(by='date')["net"].transform('sum')
    df.drop_duplicates(subset='date', inplace=True)

    trend_df = data_trend(df.copy(), "net", model='additive', period=period).to_frame("values")

    return trend_df

def all_towns_trend_year(period: int, year: int):
    df = get_data()
    #group data by town and date
    df["net"] = df.groupby(['date'])["net"].transform('sum')
    df.drop_duplicates(subset='date', inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df[df['date'].dt.year == year]
    df = df.set_index('date')
    df["date"] = df.index
    
    trend_df = data_trend(df.copy(), "net", model="additive", period=period).to_frame("values")
    
    return trend_df

def seasonality_by_town(period: int):
    df = get_data()
    #group data by town and date
    df["net"] = df.groupby(['code_town', 'date'])["net"].transform('sum')
    df.drop_duplicates(subset=['code_town', 'date'], inplace=True)

    #get towns
    towns = np.unique(df['code_town'].to_numpy())

    town_season = {}
    #get seasonality for each town
    for town in towns:
        if town != 'S001':
            town_df = df.loc[df["code_town"] == town]
            season_df = data_seasonality(town_df.copy(), "net", model="additive", period=period).to_frame("values")
            season_df["values"].fillna(season_df["values"], inplace=True)
            town_season[town] = season_df
    
    return town_season

def seasonality_by_unity(period: int):
    df = get_data()
    #group data by town and date
    df["net"] = df.groupby(['code_unity', 'date'])["net"].transform('sum')
    df.drop_duplicates(subset=['code_unity', 'date'], inplace=True)

    units = np.unique(df['code_unity'].to_numpy())
    
    unity_trend = {}
    for unity in units:
        if unity != "U00":
            unity_df = df.loc[df["code_unity"] == unity]
            trend_df = data_seasonality(unity_df.copy(), "net", model="additive", period=period).to_frame("values")
            unity_trend[unity] = trend_df
    
    return unity_trend

def all_towns_seasonality(period: int):
    df = get_data()
    #group data by town and date
    df["net"] = df.groupby(by='date')["net"].transform('sum')
    df.drop_duplicates(subset='date', inplace=True)
    seasonality_df = data_seasonality(df.copy(), "net", model='additive', period=period).to_frame("values")
    return seasonality_df

def month_clustering():
    return openPkl("models/year_cluster.pkl")

def kmeans_towns_trends():
    #extract monthly data trend for each town 
    trends = trend_by_town(365)

    #transform and normalize data
    array = []
    for key in trends.keys():
        town_df = trends[key]
        array.append(StandardScaler().fit_transform(town_df)[:,0].flatten())

    #merge data
    X_train = np.vstack(array)

    #initilize model
    model = TimeSeriesKMeans(n_clusters=6, metric="softdtw", max_iter=10)
    #fit and predict
    y_pred = model.fit_predict(X_train)
    #save model to pkl file
    model.to_pickle("models/kmeans_time_series_year.pkl")
    #model = TimeSeriesKMeans.from_pickle("models/kmeans_time_series_year.pkl")
    #y_pred = model.fit_predict(X_train)
    town_pred = {}
    for i in range(31):
        if i+1 < 10:
            town_pred["C00"+str(i+1)] = y_pred[i]
        else:
            town_pred["C0"+str(i+1)] = y_pred[i]
    
    savePkl(town_pred, "models/year_cluster.pkl")
    print(town_pred)
    #plot model results
    sz = X_train.shape[1]
    for yi in range(4):
        plt.subplot(3, 3, 4 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(model.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("DBA $k$-means")
    
    plt.show()


"""---------------Rotations--------------------"""
def rotations_trend(period: int):
    df = get_rotations_number()
    df['rotations'] = df.groupby('date')['rotations'].transform('sum')
    df.drop_duplicates('date', inplace=True)
    trend = data_trend(df, "rotations", model='additive', period=period).to_frame("values")
    
    return trend

def rotations_trend_year(period: int, year: int):
    df = get_rotations_number()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year == year]
    df['rotations'] = df.groupby('date')['rotations'].transform('sum')
    df.drop_duplicates('date', inplace=True)
    trend = data_trend(df, "rotations", model='additive', period=period).to_frame("values")
    
    return trend

def rotations_trend_by_town(period: int):
    df = get_rotations_number()
    towns = np.unique(df['code_town'].to_numpy())
    
    town_trend = {}
    for town in towns:
        if town != "S001":
            town_df = df[df["code_town"] == town]
            trend = data_trend(town_df, "rotations", model='additive', period=period).to_frame("values")
            town_trend[town] = trend
    
    return town_trend

def rotations_trend_by_town_year(town, period: int, year: int):
    df = get_rotations_number()

    town_df = df[df["code_town"] == town]
    town_df["date"] = pd.to_datetime(town_df["date"])
    town_df = town_df[town_df['date'].dt.year == year]
    town_df = town_df.set_index('date')
    town_df["date"] = town_df.index
    trend = data_trend(town_df, "rotations", model='additive', period=period).to_frame("values")
    
    return trend

def rotations_trend_by_unity(period: int):
    df = get_rotations_number()
    unities = np.unique(df['code_unity'].to_numpy())
    
    unity_trend = {}
    for unity in unities:
        if unity != "U00":
            unity_df = df[df["code_unity"] == unity]
            unity_df['rotations'] = unity_df.groupby(by='date')['rotations'].transform('sum')
            unity_df.drop_duplicates(subset='date', inplace=True)
            trend = data_trend(unity_df, "rotations", model='additive', period=period).to_frame("values")
            unity_trend[unity] = trend
    
    return unity_trend

def rotations_trend_by_unity_year(unity, period: int, year: int):
    df = get_rotations_number()

    unity_df = df[df["code_unity"] == unity]
    unity_df['rotations'] = unity_df.groupby(by='date')['rotations'].transform('sum')
    unity_df.drop_duplicates(subset='date', inplace=True)
    unity_df["date"] = pd.to_datetime(unity_df["date"])
    unity_df = unity_df[unity_df['date'].dt.year == year]
    unity_df = unity_df.set_index('date')
    unity_df["date"] = unity_df.index
    trend = data_trend(unity_df, "rotations", model='additive', period=period).to_frame("values")
    
    return trend

def rotations_trend_by_town_hour(town):
    df = get_rotations_by_hour()
    
    town_df = df[df["code_town"] == town]
    town_df['heure'] = pd.to_datetime(town_df['heure'])
    times = town_df['heure']
    town_df['rotations'] = town_df.groupby(times.dt.hour)['rotations'].transform('sum')
    town_df['heure'] = town_df['heure'].apply(lambda x: x.replace(minute=0, second=0))
    town_df.drop_duplicates(subset='heure', inplace=True)
    town_df['heure'] = town_df['heure'].dt.time
    town_df.sort_values(by='heure', ascending=True, inplace=True)
    
    return town_df

def rotations_trend_by_unity_hour(unity):
    df = get_rotations_by_hour()
        
    unity_df = df[df["code_unity"] == unity]
    unity_df['heure'] = pd.to_datetime(unity_df['heure'])
    times = unity_df['heure']
    unity_df['rotations'] = unity_df.groupby(times.dt.hour)['rotations'].transform('sum')
    unity_df['heure'] = unity_df['heure'].apply(lambda x: x.replace(minute=0, second=0))
    unity_df.drop_duplicates(subset='heure', inplace=True)
    unity_df['heure'] = unity_df['heure'].dt.time
    unity_df.sort_values(by='heure', ascending=True, inplace=True)
    unity_df.dropna(axis=0, inplace=True)
    
    return unity_df

def rotations_trend_by_hour():
    df = get_rotations_by_hour()
    df['heure'] = pd.to_datetime(df['heure'])
    df['heure'] = df['heure'].apply(lambda x: x.replace(minute=0, second=0))
    df['rotations'] = df.groupby(by='heure')['rotations'].transform('sum')
    df.drop_duplicates(subset='heure', inplace=True)
    df['heure'] = df['heure'].dt.time
    df.sort_values(by='heure', ascending=True, inplace=True)
    df.dropna(axis=0, inplace=True)
    
    return df



def main():
    df = get_data()
    print(trend_by_town_year('C002', 30, 2021))
    #model = TimeSeriesKMeans.from_pickle("models/kmeans_time_series.pkl")
    #print(model)


if __name__ == "__main__":
    main()