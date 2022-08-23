from selenium import webdriver
import numpy as np
import time
import pandas as pd


browser = webdriver.Chrome()
url = "https://fminside.net/clubs"
browser.get(url)
dict_name_link = {"CName": [], "CLink": []}
#liste = []
club = browser.find_elements_by_xpath("//span[@class='name']/b/a")
for i in club:
    dict_name_link["CName"].append(i.text)
    dict_name_link["CLink"].append(i.get_attribute('href'))

data = pd.DataFrame(dict_name_link)
asd = {"CName": [], "CCity": [], "CLeague" : [], "CReputation" : [], "CBalance" :[], "CTransfer_Budget" : [], "CTotal_Wages" : [],"CRemaining_Wages" : []}
asd["CName"] = dict_name_link["CName"]


for i in dict_name_link["CLink"]:
    browser.get(i)

    city = browser.find_element_by_xpath("//*[@id='club']/div/div[1]/ul/li[3]/span[2]")
    asd["CCity"].append(city.text)

    League = browser.find_element_by_xpath("//*[@id='club']/div/div[1]/ul/li[6]/span[2]")
    asd["CLeague"].append(League.text)

    Reputation = browser.find_element_by_xpath("//*[@id='club']/div/div[1]/ul/li[7]/span[2]")
    asd["CReputation"].append(Reputation.text)

    Balance = browser.find_element_by_xpath("//*[@id='club']/div/div[2]/ul/li[1]/span[2]")
    asd["CBalance"].append(Balance.text)

    Transfer_Budget =browser.find_element_by_xpath("//*[@id='club']/div/div[2]/ul/li[2]/span[2]")
    asd["CTransfer_Budget"].append(Transfer_Budget.text)

    Total_Wages= browser.find_element_by_xpath("//*[@id='club']/div/div[2]/ul/li[2]/span[2]")
    asd["CTotal_Wages"].append(Total_Wages.text)

    Remaining_Wages = browser.find_element_by_xpath("//*[@id='club']/div/div[2]/ul/li[3]/span[2]")
    asd["CRemaining_Wages"].append(Remaining_Wages.text)

browser.close()

club_df = pd.DataFrame(asd)
a=0
for i,j,k,l in zip(club_df["CBalance"],club_df["CTransfer_Budget"],club_df["CTotal_Wages"],club_df["CRemaining_Wages"]) :
    i = i.replace("€","")
    j = j.replace("€","")
    k = k.replace("€","")
    l = l.replace("€","")
    l = l.replace("pw","")
    club_df["CBalance"][a] = i
    club_df["CTransfer_Budget"][a] = j
    club_df["CTotal_Wages"][a] = k
    club_df["CRemaining_Wages"][a] = l
    a+=1


