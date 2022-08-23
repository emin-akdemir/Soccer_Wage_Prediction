import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve, \
    train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from folium import GeoJson 
import folium
from folium.plugins import heat_map,MarkerCluster
from sklearn.ensemble import VotingRegressor





df_players = pd.read_csv("fm22_players_15k.csv")
df_clubs = pd.read_csv("fm22_clubs_3k.csv")

def all_values():
    df = pd.concat([df_players,df_clubs],axis=1)
    return df

df = all_values()


#df = pd.concat([df_players,df_clubs],axis=1)



def na(df):
    na_values = pd.isnull(df).sum().sort_values(ascending=False)
    return pd.DataFrame(na_values)

#na(df)


def data_prep(df,df_clubs):
    print("Data Preparation ...")
    ########################################
    # Veri Ön İşleme
    ########################################
    # Külüp datasındaki kolonları ana data setine daha sonra doldurmak üzere
    # boş değer ataması yaparak ekledik
    #  Oyuncuların kuluplerine bakarak club datasetindeki bilgileri oyuncu özelinde
    # atama işlemi yaptık
    df_clubs["CLeague"] = df_clubs["CNation"] + " " + df_clubs["CLeague"]
    df[df_clubs.columns] = np.nan  #birleştirdğimiz verinin club columnslarını nan yaptık.


    for i,col in enumerate(df["Team"]):
        for j,col_club in enumerate(df_clubs["CName"]):
            if col == col_club:
                df.iloc[i,55:] = df_clubs.loc[j]
                break


    # çiftleme verileri silme
    df = df[~df["Link"].duplicated()]

    # Veri setinden bulunan "None" değerlerine boş değer ataması yaptık
    df = df.replace(to_replace="None", value=np.nan)


    # Release_Clause ---> Çok fazla boş değer bulunduğundan ötürü veri setinden çıkarttık.
    # Diğer kolonlar model açısından bilgi içermiyeceğinden ötürü çıkarttık

    df_name_Cname = df[["Name","CName","Img_Link"]]
    df.drop(["Name","Unnamed: 0",'Link',"Img_Link","Release_Clause","CName","CLink",'Unique_ID',"CCity","CStatus"], axis=1,inplace=True)

    #############################
    # Boş değer doldurma
    ############################

    # Hedef değişkende bulunan boş değerler silindi
    df = df[df["Wages"].notna()] #notna True False verir df[df["Wages"].notna()] dediğimizde sadece True değerler vererek oto silmiş olmuş. df = df.dropna(subset = ["Wages"]) de kullanabilriz.  

    # Sell_value(Modeli kötü etkiliyebilir) ! Kötü bir futbolcuyu median'a göre doldurmuş olabilir
    df.loc[df["Sell_Value"] == "Not for sale","Sell_Value"] = np.nan
    df.loc[~df["Sell_Value"].isnull(),"Sell_Value"] = df[~df["Sell_Value"].isnull()]["Sell_Value"].str[2:].str.replace(",", "").astype(float)
    df["Sell_Value"].fillna((df.loc[~df["Sell_Value"].isnull(),"Sell_Value"].median()),inplace=True)

    # Potential 22
    # İş bilgisine dayanarak(Wonderkid futbolcular) ve sitede bulunan bilgilendirmeler ile  Potensiyel gücü eksik olan veriler dolduruldu
    df.loc[(df["Potential"].isnull()) & (df["Ability"] >= 70), "Potential"] = 78
    df.loc[(df["Potential"].isnull()) & (df["Ability"] < 70), "Potential"] = 64
    #df["Potential"].loc[df["Ability"] < 70].median()

    #Geri Kalanları sildik
    df.dropna(inplace=True)

    #########################
    # Veri düzeltme
    #########################

    # Length
    df["Length"] = df["Length"].str[:3]
    df["Length"] = df["Length"].astype(int)
    # Weight
    df["Weight"] = df["Weight"].str[:2]
    df["Weight"] = df["Weight"].astype(int)
    #Wages
    df["Wages"] = df["Wages"].str[1:-2].str.replace(",", "").astype(float)
    # Contract_End
    df["Contract_End"] = df["Contract_End"].apply(pd.to_datetime)
    df['Contract_End_Year'] = df["Contract_End"].dt.year

    # CBalance
    df.loc[df["CBalance"].str.contains("K") == True,"CBalance"] = df[df["CBalance"].str.contains("K") == True]["CBalance"].str[1:-1].astype(float) * 1000
    df.loc[df["CBalance"].str.contains("M") == True,"CBalance"] = df[df["CBalance"].str.contains("M") == True]["CBalance"].str[1:-1].astype(float) * (10**6)
    df["CBalance"] = df["CBalance"].astype(float)
    # CTransfer_Budget
    df.loc[df["CTransfer_Budget"].str.contains("K") == True,"CTransfer_Budget"] = df[df["CTransfer_Budget"].str.contains("K") == True]["CTransfer_Budget"].str[1:-1].astype(float) * 1000
    df.loc[df["CTransfer_Budget"].str.contains("M") == True,"CTransfer_Budget"] = df[df["CTransfer_Budget"].str.contains("M") == True]["CTransfer_Budget"].str[1:-1].astype(float) * (10**6)
    df["CTransfer_Budget"] = df["CTransfer_Budget"].astype(float)

    # CTotal_Wages
    df["CTotal_Wages"] = df["CTotal_Wages"].str[:-2]
    df["CTotal_Wages"] = df["CTotal_Wages"].str.strip()
    df.loc[df["CTotal_Wages"].str.contains("K") == True,"CTotal_Wages"] = df[df["CTotal_Wages"].str.contains("K") == True]["CTotal_Wages"].str[1:-1].astype(float) * 1000
    df.loc[df["CTotal_Wages"].str.contains("M") == True,"CTotal_Wages"] = df[df["CTotal_Wages"].str.contains("M") == True]["CTotal_Wages"].str[1:-1].astype(float) * (10**6)
    df["CTotal_Wages"] = df["CTotal_Wages"].astype(float)
    # CRemaining_Wages
    df["CRemaining_Wages"] = df["CRemaining_Wages"].str[:-2]
    df["CRemaining_Wages"] = df["CRemaining_Wages"].str.strip()
    df.loc[df["CRemaining_Wages"].str.contains("K") == True,"CRemaining_Wages"] = df[df["CRemaining_Wages"].str.contains("K") == True]["CRemaining_Wages"].str[1:-1].astype(float) * 1000
    df.loc[df["CRemaining_Wages"].str.contains("M") == True,"CRemaining_Wages"] = df[df["CRemaining_Wages"].str.contains("M") == True]["CRemaining_Wages"].str[1:-1].astype(float) * (10**6)
    df["CRemaining_Wages"] = df["CRemaining_Wages"].astype(float)
    # CFounded
    df["CFounded"] = pd.to_numeric(df["CFounded"],errors='coerce')
    # CMost_Talented_XI
    df["CMost_Talented_XI"] = df["CMost_Talented_XI"].astype(int)

    # Oyuncuların pozisyon bilgisinin sınıflarını birleştirdik
    df.loc[((df['Position'].str.contains("ST")) | (df['Position'].str.contains("AMR")) | (df['Position'].str.contains("AML"))), "Position"] = "Striker"
    df.loc[((df['Position'].str.contains("DM")) | (df['Position'].str.contains("ML")) | (df['Position'].str.contains("MC")) | (df['Position'].str.contains("MR")) | (df['Position'].str.contains("AMC"))), "Position"] = "Midfield"
    df.loc[((df['Position'].str.contains("DL")) | (df['Position'].str.contains("DR")) | (df['Position'].str.contains("DC")) | (df['Position'].str.contains("WBL")) | (df['Position'].str.contains("WBR"))), "Position"] = "Defenders"
    ############################
    # Filtreleme
    df = df.loc[df["Wages"] < 45250.5] ##maaşı 50k üstü oyuncuları almıyoruz değerlendirmeye

    return df,df_name_Cname



#############  VERİ GÖRSELLEŞTİRME ###################
def visualization(df):
    
    # Potential & Wage
    plt.figure(figsize=(7,5))
    ax = sns.scatterplot(x=df["Potential"], y=df["Wages"])
    plt.xlabel("Potential")
    plt.ylabel("Wages")
    plt.title("Potential & Wage", fontsize = 18)
    plt.savefig("Potential&Wage_Dağılımı.png")
    plt.show()

    # Potential & Sell_Value
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(x =df['Potential'], y = df['Sell_Value'])
    plt.xlabel("Potential")
    plt.ylabel("Sell_Value")
    plt.title("Potential & Sell_Value", fontsize = 18)
    plt.savefig("Potential&Sell_Value_Dağılımı.png")
    plt.show()

    # Ayak durumuna göre potential&Wage dağılımı
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(x =df['Potential'], y = df['Wages'], hue = df['Foot']) #hue = futbolcuların hangi ayak kullanıdıgına göre grafikte ayrı gösterir.
    plt.xlabel("Potential")
    plt.ylabel("Wage")
    plt.title("Foot & Potential & Wage", fontsize = 18)
    plt.savefig("Ayak_Potential&Wage_dağılımı.png")
    plt.show()
    
    # Wordcloud

    df.Nation.loc[df.Nation.isnull()] = str('NaN')   #nan değerler float olarak gözküyordu ve hata veriyordu str ye çevirdik.
    text = " ".join(i for i in df.Nation)
    wordcloud = WordCloud(collocations=False).generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("Nation_WordCloud.png")
    plt.show()

    #Wage Sell_Value Dağılımları
    plt.figure(figsize=(7,5))
    ax = sns.scatterplot(x =df['Sell_Value'], y = df['Wages'])
    plt.savefig("Wage_and_Sell_Value_Dağılımları.png")
    plt.show()

    # Dünya haritası ile görselleştirme
    df_geo = df[["CNation","CCity"]]
    df_geo["Long"] = np.nan
    df_geo["Lat"] = np.nan
    geolocator = Nominatim(user_agent="my_user_agent")
    #df["Nation"].value_counts()  #Hangi ülkeden kaç değer var.
    #df["Name"].loc[df["Nation"] == "Turkey"] #Nation ı Turkey olan futbolcuların ismi.
    lst_nation = df_geo["CNation"].value_counts().index.tolist() #Ülkeleri indexe göre liste yapar. !Not: İlk sıraya value_counts() dan dolayı en çok hangi ülkeden oyuncu varsa onu koyar.
    lst_city = df_geo["CCity"].value_counts().index.tolist()
    dict_nation={}
    for i in lst_city:
        loc = geolocator.geocode(i)
        if loc:
            dict_nation.update({i:[loc.longitude,loc.latitude]})
        else:
            print(i)
            dict_nation.update({i:[37.1833,67.3667]})
    
    df_geo["Potential"] = df["Potential"]
    df_geo["Wages"] = df["Wages"]
    df_geo["Potential_Mean"] = np.nan
    df_geo["Wages_Mean"] = np.nan
    for i in lst_city :
        df_geo.loc[df_geo["CCity"] == i, "Long"] = dict_nation[i][0]
        df_geo.loc[df_geo["CCity"] == i, "Lat"] = dict_nation[i][1]
    
    for j in lst_nation:
        df_geo.loc[df_geo["CNation"] == j, "Potential_Mean"] = df_geo[df_geo["CNation"] == j]["Potential"].mean()
    for j in lst_nation:
        df_geo.loc[df_geo["CNation"] == j, "Wages_Mean"] = df_geo[df_geo["CNation"] == j]["Wages"].mean()
    
    df_geo[["Name","CName","Img_Link"]]= df
    df_geo[["Ability","Age","Foot","Position","Caps_Goals","Length","Weight","Nation"]] = df[["Ability","Age","Foot","Position","Caps_Goals","Length","Weight","Nation"]]
    
    df_geo.to_csv("df_geo.csv")

    # Verileri Mape Yerleştirme
    df_geo = pd.read_csv("df_geo.csv")
    geo=r"archive/countries.geojson"
    file = open(geo, encoding="utf8")
    text = file.read()
    # Futbolcu potansiyellerine göre dağılmını map üzerinde gösterilmesi
    m = folium.Map([42, 29], tiles="Cartodb Positron", zoom_start=5, width="%100", height="%100")
    folium.Choropleth(
        geo_data=text,
        data=df_geo,
        columns=['CNation', 'Potential_Mean'],
        legend_name='Oynadıkları liglere göre Potansiyel Oyuncu Dağılımı',
        key_on='feature.properties.ADMIN'
    ).add_to(m)
    m.save('Potensiyel_ortalamasına_göre_club_ülke_dağılımı.html')

    m = folium.Map([42, 29], tiles="Cartodb Positron", zoom_start=5, width="%100", height="%100")
    folium.Choropleth(
        geo_data=text,
        data=df_geo,
        columns=['Nation', 'Potential_Mean'],
        legend_name='Oynadıkları liglere göre Potansiyel Oyuncu Dağılımı',
        key_on='feature.properties.ADMIN'
    ).add_to(m)
    m.save('Potensiyel_ortalamasına_göre_ülke_dağılımı.html')


    # Futbolcuların ülkelere göre dağılımını HeatMap olarak  gösterilmesi
    m=folium.Map(location=[40,32],tiles="OpenStreetMap",zoom_start=7)
    folium.plugins.HeatMap(zip(df_geo["Lat"],df_geo["Long"])).add_to(m)
    m.save('Nation_Heatmap.html')

    # Futbolcuların şehirlere göre göre dağılımını MarkerCluster gösterilmesi
    m=folium.Map(location=[40,32],tiles="OpenStreetMap",zoom_start=7)
    folium.plugins.MarkerCluster(zip(df_geo["Lat"],df_geo["Long"])).add_to(m)
    m.save('Nation_MarkerCluster.html')

    # Oyuncuların oyanığı takımlara göre dağılımı ve bilgilerinin gösterilmesi
    m3 = folium.Map(location=[40, 32], tiles="OpenStreetMap", zoom_start=7)
    marker = MarkerCluster().add_to(m3)
    for i in df_geo.index:
        iframe = folium.IFrame("<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<h3><b> Name: </b></font>' + str(df_geo.loc[i, 'Name']) + '</h3><br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Nation: </b></font>' + str(df_geo.loc[i, 'Nation']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Ability: </b></font>' + str(df_geo.loc[i, 'Ability']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Potential: </b></font>' + str(df_geo.loc[i, 'Potential']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Age: </b></font>' + str(df_geo.loc[i, 'Age']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Position: </b></font>' + str(df_geo.loc[i, 'Position']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Foot: </b></font>' + str(df_geo.loc[i, 'Foot']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Length: </b></font>' + str(df_geo.loc[i, 'Length']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Weight: </b></font>' + str(df_geo.loc[i, 'Weight']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Caps_Goals: </b></font>' + str(df_geo.loc[i, 'Caps_Goals']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Wages: </b></font>' + str(df_geo.loc[i, 'Wages']))
        popup = folium.Popup(iframe, min_width=300, max_width=300)
        # lat=df_geo.loc[i,"Lat"]+np.random.uniform(0.1, 10**(-20))-0.00005
        # long=df_geo.loc[i,"Long"]+np.random.uniform(0.1, 10**(-20))-0.00005
        folium.Marker(location=[df_geo.loc[i, "Lat"], df_geo.loc[i, "Long"]], popup=popup, marker_cluster=True,
                      icon=folium.DivIcon(html=f"""<div><img src='""" + df_geo.loc[
                          i, "Img_Link"] + """' width="300%" height="300%"></div>""")).add_to(marker)
    marker.save('Image_Map.html')


def corr_anlys(df):
    df.columns
    df["Wages"] = df["Wages"].str[2:-2].str.replace(",","").astype(float)
    cols = df.corr()["Wages"].sort_values(ascending=False)

def feature_eng(df):
    print("Feature Engineering ...")
    #########################################
    # Feature Engineering
    ###################
    #Değişken Üretme
    ###################
    # Futbolcuların sözleşme bitiş tarihini kullanarak yeni değişkenler oluşturuldu.
    today_date = dt.datetime(2022, 1, 1)
    df['Contract_End'] = pd.to_datetime(df['Contract_End']) #Bunu yazmazsak .dt hatası alıyoruz. dtype ı değiştirdik bu kodu kullanarak
    df['Contrat_end_month'] = df["Contract_End"].dt.month
    df['Contrat_end_day'] = df["Contract_End"].dt.day
    df['Contrat_end_year'] = df["Contract_End"].dt.year
    df["Contrat_end_left_days"] = (df["Contract_End"]-today_date).dt.days
    df["Contrat_end_left_year"] = (df["Contract_End"].dt.year-today_date.year)
    df["Contrat_end_left_month"] = (df["Contract_End"]-today_date)/np.timedelta64(1,"M")
    df.drop("Contract_End",axis=1,inplace=True)

    # Oyuncuların yaşlarına ve potensiyel güçlerine göre sınıfladırma yapılarak yeni değişken üretildi
    #df["Age_Potential_Seg"] = ""
    #df.loc[(df["Age"] <= 20) & (df["Potential"] >= 80), "Age_Potential_Seg"] = "Wonderkid"
    #df.loc[(df["Age"] <= 20) & (df["Potential"] < 80), "Age_Potential_Seg"] = "Tecrübesiz"
    #df.loc[(df["Age"] > 20) & (25 >= df["Age"]) & (df["Potential"] >= 80), "Age_Potential_Seg"] = "Star_Candidate"
    #df.loc[(df["Age"] > 20) & (25 >= df["Age"]) & (df["Potential"] < 80), "Age_Potential_Seg"] = "Developers"
    #df.loc[(df["Age"] > 25) & (35 >= df["Age"]) & (df["Potential"] >= 80), "Age_Potential_Seg"] = "Star"
    #df.loc[(df["Age"] > 25) & (35 >= df["Age"]) & (df["Potential"] < 80), "Age_Potential_Seg"] = "Star-"
    #df.loc[(35 < df["Age"]) & (df["Potential"] >= 80), "Age_Potential_Seg"] = "Star+"
    #df.loc[(35 < df["Age"]) & (df["Potential"] < 80), "Age_Potential_Seg"] = "Çöp"

    #  Yeni değişkenler
    df["Ability_Potential"] = df["Ability"] * df["Potential"]
    df["New_Most_Best"] = df["CBest_XI"] - df["CMost_Talented_XI"]
    df["New_Rep_Best_Tal"] = df["CBest_XI"] * df["CMost_Talented_XI"] * df["CReputation"]
    df["New_Tack_Mark"] = df["Tackling"] + df["Marking"]
    df["New_Pos_Mark"] = df["Positioning"] * df["Marking"]
    df["New_Jump_Leng"] = df["Length"] / df["Jumping_Reach"]

    # Caps değişkeni iki değişkene ayırıldı(veri önişlemeye alınabilir)
    df[['Caps', 'Goals']] = df['Caps_Goals'].str.split('/', expand=True)
    df.drop("Caps_Goals", axis=1, inplace=True)
    df["Caps"] = df["Caps"].astype(int)
    df["Goals"] = df["Goals"].astype(int)

    #Sözel Kısımları sayısal değerlere çevirme
    #Encoding
    labelencoder = LabelEncoder()
    df["Nation"] = labelencoder.fit_transform(df["Nation"])
    df["Team"] = labelencoder.fit_transform(df["Team"])
    df["Position"] = labelencoder.fit_transform(df["Position"])
    df["Foot"] = labelencoder.fit_transform(df["Foot"])
    df["CLeague"] = labelencoder.fit_transform(df["CLeague"])
    df["CNation"] = labelencoder.fit_transform(df["CNation"])

    
    


value = {"name": [], "test_hatasi": [],"Doğruluk Oranı" : []}
models = [("LightGBM",LGBMRegressor()),
          ("XGBoost",XGBRegressor()),
          ("RF",RandomForestRegressor()),
          ("Decs",DecisionTreeRegressor()),
          ("CatBoost",CatBoostRegressor()),
          ("GBM",GradientBoostingRegressor())]
    
def modelleme(df,alg,x_train,x_test, y_train, y_test):
    print("Modelleme...")
    for name,alg in models:
        #train-test ayrımı
        #df["Ability"].loc[df["Ability"].isnull()] = 70
        #df["Wages"].loc[df["Wages"].isnull()] = 50000
        y = df["Wages"]
        x = df.drop(["Wages"],axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
        
        model = alg.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        accuracy = model.score(x_test,y_pred)
        RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
        value["name"].append(name)
        value["test_hatasi"].append(RMSE)
        value["Doğruluk Oranı"].append(accuracy)


    #chart = sns.barplot(data = model_kiyaslama ,
    #                x = 'name',
    #                y='test_hatasi')

    #identity_line = np.linspace(max(min(y_pred), min(y_test)),
    #                                        min(max(y_pred), max(y_test)))
    #plt.figure(figsize=(10, 10))
    #plt.scatter(x=y_pred, y=y_test, alpha=0.2)
    #plt.plot(identity_line, identity_line, color="red", linestyle="dashed", linewidth=3.0)
    #plt.show()
    #plt.savefig("Gerçek_tahmin_Dağılımı.png")

def best_model(df,alg,x_train,x_test, y_train, y_test):
    modelleme(df,alg,x_train,x_test, y_train, y_test)
    print("Best Modelleme...")

    best_model = alg().fit(x_train,y_train)
    y_pred = best_model.predict(x_test)

    #Model Tuning
    lgb_params = {"learning_rate": [0.01,0.1,0.5,1],
              "n_estimators": [20,40,100,200,500,1000],
              "max_depth": [1,2,3,4,5,6,7,8,9,10]}
    #lgm_cv_model = GridSearchCV(best_model,lgb_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)
    #print(lgm_cv_model.best_params_)

def hyperparameter_optimization(df,alg,x_train,x_test, y_train, y_test):
    best_model(df,alg,x_train,x_test, y_train, y_test)
    print("Hiperparametre...")

    tuned_model = CatBoostRegressor().fit(x_train,y_train)
    y_pred = tuned_model.predict(x_test)
    print("TUNED MSE:",np.sqrt(mean_squared_error(y_test,y_pred)))
    accuracy2 = tuned_model.score(x_test,y_pred) 
    print("Score:",accuracy2)
    
def voting_regression(models, X_train, y_train):
    print("Voting Regression...")
    voting_clf = VotingRegressor(estimators=[(models[0][0], models[0][1]), (models[1][0], models[1][1]),
                                              (models[2][0], models[2][1])]).fit(X_train, y_train)

    cv_results = cross_validate(voting_clf, X_train, y_train, cv=5,
                                scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])
    print("VR RMSE   : ", -(cv_results['test_neg_root_mean_squared_error'].mean()))
    print("VR MAE    : ", -(cv_results['test_neg_mean_absolute_error'].mean()))
    print("VR R-KARE :", (cv_results['test_r2'].mean()))
    return voting_clf


def main():
    df = all_values()
    df,df_name_Cname = data_prep(df,df_clubs)
    feature_eng(df)
    y = df["Wages"]
    x = df.drop(["Wages"],axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    hyperparameter_optimization(df,CatBoostRegressor,x_train,x_test,y_train, y_test)
    print(pd.DataFrame(value).sort_values(by="test_hatasi"))

    voting_clf = voting_regression(models,x_train,y_train)
    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf


if __name__ == "__main__":
    print("İşlem başladı")
    #main()


import streamlit as st
import streamlit.components.v1 as components
import base64

st.set_page_config(layout="wide")
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('blue_bg.png')  
# lightgbm==3.3.2
# catboost==1.0.5
# xgboost==1.2.0
# gbm==0.0.1
# random-forest-mc==0.3.7
df = pd.read_csv(r"./final_1.csv")
df_2 = pd.read_csv(r"./final_2.csv")
df_X_test = pd.read_csv(r"./xtest.csv")
####################
### INTRODUCTION ###
####################

row0_spacer1, row0_1, row0_spacer2, row0_spacer3 = st.columns((.1, 2.3, .1, .1))
with row0_1:
    st.markdown(f'<h1 style="color:#EC350F;font-size:40px;">{"Football Players Wage Prediction"}</h1>', unsafe_allow_html=True)
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown(f'<h1 style="color:#EC350F;font-size:25px;">{"Arda Savaş - Emin Akdemir"}</h1>', unsafe_allow_html=True)

### SEE DATA ###
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader("Data:")

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
with row2_1:
    unique_players_in_df = df_2.Name.nunique()
    str_players = "⚽ " + str(unique_players_in_df) + " Players"
    st.markdown(f'<h1 style="color:#FFFFFF;font-size:20px;">{str_players}</h1>', unsafe_allow_html=True)
with row2_2:
    unique_teams_in_df = len(np.unique(df_2.Team).tolist())
    t = " Teams"
    if(unique_teams_in_df==1):
        t = " Team"
    str_teams = "🏟️" + str(unique_teams_in_df) + t
    st.markdown(f'<h1 style="color:#FFFFFF;font-size:20px;">{str_teams}</h1>', unsafe_allow_html=True)
with row2_3:
    total_league_in_df = len(np.unique(df_2.CLeague).tolist())
    str_league = "👟⚽" + str(total_league_in_df) + " League"
    st.markdown(f'<h1 style="color:#FFFFFF;font-size:20px;">{str_league}</h1>', unsafe_allow_html=True)

row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
with row3_1:
    st.markdown("")
    see_data = st.expander('You can click here to see the raw data first 👉')
    with see_data:
        st.dataframe(data=df_2.iloc[:,:48].reset_index(drop=True))
st.text('')

############################
# Model deployment
###########################

### TEAM SELECTION ###
#unique_teams = get_unique_teams(df_data_filtered_matchday)
st.title("Wage Prediction")
all_nation_selected = st.selectbox("Nation of the Player's League ",
                                           [""]+df_2.CNation.value_counts().index.tolist(),
                                           format_func=lambda x: "" if x == "" else x)

all_league_selected = st.selectbox("Player's League",
                                           [""]+df_2[df_2.CNation == all_nation_selected]["CLeague"].value_counts().index.tolist(),
                                           format_func=lambda x: "" if x == "" else x)

all_teams_selected = st.selectbox("Player's Team",
                                          [""]+df_2[df_2.CLeague == all_league_selected]["Team"].value_counts().index.tolist(),
                                          format_func=lambda x: "" if x == "" else x)

all_player_selected = st.selectbox("Player's Name",
                                           [""]+df_2[df_2.Team == all_teams_selected]["Name"].value_counts().index.tolist(),
                                           format_func=lambda x: "" if x == "" else x)


def fm22_prediction(df, name):
    if name == "":
        return ""
    else:
        index = df[df["Name"] == str(name)].index.tolist()[0]
        final_df = pd.read_csv(r"./final_1.csv")
        model = joblib.load(r'./voting_clf.pkl')
        y = final_df["Wages"]
        X = final_df.drop(["Wages","Name"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
        return model.predict(X.iloc[index].values.reshape(1, -1))  

wage = fm22_prediction(df, str(all_player_selected))

try:
    rw = df[df["Name"] == all_player_selected]["Wages"].tolist()[0]
    pw = wage[0]
    st.write("Prediction Wage **€ {:.2f}** pw".format(pw))
    st.write("Real Wage **€ {:.2f}** pw".format(rw))

    ps = 100 * (abs(rw - pw) / rw)
    st.write("Predicted Accuracy Rate Error: % {:.2f}".format(ps))
except:
    st.write("Player for Prediction")



################
### ANALYSIS ###
################

### DATA EXPLORER ###
row12_spacer1, row12_1, row12_spacer2 = st.columns((.2, 7.1, .2))
with row12_1:
    st.subheader('Player Information')

row13_spacer1, row13_1, row13_spacer2, row13_2, row13_spacer3, row13_3, row13_spacer4, row13_4, row13_spacer4 = st.columns((.2, 2.3, .2, 2.3, .2, 2.3, .2, 2.3, .2))
with row13_1:
    show_me_Nation = st.selectbox("Nation of the Player's League",
                                df_2.CNation.value_counts().index.tolist(),
                                key="playernation")

with row13_2:
    show_me_league = st.selectbox("Player's League",
                                 df_2[df_2.CNation == show_me_Nation]["CLeague"].value_counts().index.tolist(),
                                 key="playerleague")

with row13_3:
    show_me_team = st.selectbox("Player's Team",
                                df_2[df_2.CLeague == show_me_league]["Team"].value_counts().index.tolist(),
                                key="playerteam")

with row13_4:
    show_me_player = st.selectbox("Player's Name",
                                  df_2[df_2.Team == show_me_team]["Name"].value_counts().index.tolist(),
                                  key="playername")


#row15_spacer1, row15_1, row15_2, row15_3, row15_4, row15_spacer2  = st.columns((0.5, 1.5, 1.5, 1, 2, 0.5))
row15_spacer1, row15_1, row15_spacer2, row15_2, row15_spacer3, row15_3, row15_spacer4, row15_4, row15_spacer4 = st.columns((.2, 2.3, .2, 2.3, .2, 2.3, .2, 2.3, .2))
with row15_1:
    st.subheader("Player")
with row15_2:
    st.subheader("Technical")
with row15_3:
    st.subheader("Mental")
with row15_4:
    st.subheader("Physical")

row16_spacer1, row16_1, row16_spacer2, row16_2, row16_spacer3, row16_3, row16_spacer4, row16_4, row16_spacer4 = st.columns((.2, 2.3, .2, 2.3, .2, 2.3, .2, 2.3, .2))
#row16_spacer1, row16_1, row16_2, row16_3, row16_4, row16_spacer2= st.columns((0.5, 1.5, 1.5, 1, 2, 0.5))
with row16_1:
    col_player_info = df_2.columns[:10]
    k=0
    for i in df_2[col_player_info]:
        if k==0:
            st.image(df_2[df_2["Name"] == str(show_me_player)].loc[:,"Img_Link"].tolist()[0],width=125)

        st.markdown('**' + i + '**: ' + '' + str(df_2[col_player_info][df_2["Name"] == str(show_me_player)].loc[:, i].tolist()[0]) + '')
        k=k+1
with row16_2:
    col_player_tech = df_2.columns[10:24]
    for i in df_2[col_player_tech]:
        st.markdown('**' + i + '**: ' + '' + str(df_2[col_player_tech][df_2["Name"] == str(show_me_player)].loc[:, i].tolist()[0]) + '')

with row16_3:
    col_player_mental = df_2.columns[24:38]
    for i in df_2[col_player_mental]:
        st.markdown('**' + i + '**: ' + '' + str(df_2[col_player_mental][df_2["Name"] == str(show_me_player)].loc[:, i].tolist()[0]) + '')

with row16_4:
    col_player_physical = df_2.columns[38:46]
    for i in df_2[col_player_physical]:
        st.markdown('**' + i + '**: ' + '' + str(df_2[col_player_physical][df_2["Name"] == str(show_me_player)].loc[:, i].tolist()[0]) + '')

st.text("")
st.text("")
st.subheader("Mapping Players by Team")
st.write("The Cities of the Players' Team visualized on the World Map")
html = open(r"./Image_Map.html",'r',encoding='utf-8')
source = html.read()
components.html(source,width=1350,height=500)

#4C78A9
######################

