import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
import joblib
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
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
st.write("The Cities of the Players' Team visualized on the map Map")
html = open(r"./Image_Map.html",'r',encoding='utf-8')
source = html.read()
components.html(source,width=1350,height=500)

#4C78A9
######################

