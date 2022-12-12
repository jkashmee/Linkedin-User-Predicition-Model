#Imports
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.markdown("# LinkedIn User Prediction Model")
st.markdown("### Created by: Jon Kashmeeri")
st.markdown("#### The purpose of this survey is to predict the probability of a person being a LinkedIn user. Please answer the questions below to view the predicted user.")


#User Input
par = st.selectbox("Are you a parent of a child under 18?",
        options = ["Yes",
                    "No"])

if par == "Yes":
    par = 1
else:
    par = 0


mar = st.selectbox("Are you married",
options = ["Yes",
            "No"])

if mar == "Yes":
    mar = 1
else:
    mar = 0


gen = st.selectbox("What is your gender?",
options = ["Female",
            "Male"])

if gen == "Female":
    gen = 2
else:
    gen = 0


inc = st.selectbox("What is your income?",
options = ["Less than 10,000",
            "10 to under 20,000",
            "20 to under 30,000",
            "30 to under 40,000",
            "40 to under 50,000",
            "50 to under 75,000",
            "75 to under 100,000",
            "100 to under 150,000",
            "150,000 or more"])

if inc == "Less than 10,000":
    inc = 1
if inc == "10 to under 20,000":
    inc = 2
if inc == "20 to under 30,000":
    inc = 3
if inc == "30 to under 40,000":
    inc = 4
if inc == "40 to under 50,000":
    inc = 5
if inc == "50 to under 75,000":
    inc = 6
if inc == "75 to under 100,000":
    inc = 7
if inc == "100 to under 150,000":
    inc = 8
if inc == "150,000 or more":
    inc = 9


edu = st.selectbox("What is your education level?",
options = ["Less than high school",
            "Partial high school",
            "High school graduate",
            "Some college (no degree)",
            "Two-year college degree",
            "Four-year college degree",
            "Partial postgraduate",
            "Postgraduate degree"])

if edu == "Less than high school":
    edu = 1
if edu == "Partial high school":
    edu = 2
if edu == "High school graduate":
    edu = 3
if edu == "Some college (no degree)":
    edu = 4
if edu == "Two-year college degree":
    edu = 5
if edu == "Four-year college degree":
    edu = 6
if edu == "Partial postgraduate":
    edu = 7
if edu == "Postgraduate degree":
    edu = 8


age = st.slider(label = "Enter your age (select max value for 97+)",
    min_value = 0,
    max_value = 97,
    value = 0)

#Prediction model

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where((x == 1),1,0)  
    return x

ss = pd.DataFrame(
{
    "parent":np.where(s["par"]==1,1,0),
    "married":np.where(s["marital"]==1,1,0),
    "female":np.where(s["gender"]==2,1,0),
    "income":np.where(s["income"]>9,0,s["income"]),
    "education":np.where(s["educ2"]>8,0,s["educ2"]),
    "age":np.where(s["age"]>=97,0,s["age"]),
    "sm_li":np.where(clean_sm(s["web1h"])==1,1,0)
})

y = ss["sm_li"]
x = ss[["parent","married","female","income","education","age"]]

X_train,X_test, Y_train,Y_test = train_test_split(x,y,
                                            stratify = y,
                                            test_size = .2,
                                            random_state = 8162001)

lr = LogisticRegression()
lr.fit(X_train,Y_train)

user = [par,mar,gen,edu,inc,age]
predicted_class = lr.predict([user])
probs = lr.predict_proba([user][0][1])

userclass=""
if predicted_class[0] == 1:
    userclass = "User is a LinkedIn user"
else:
    userclass = "User is not a LinkedIn user"


st.markdown("### Results")

st.write(f"Prediction: {userclass}")
st.write(f"Probability of model [being a user, not being a user]   =   {probs[0]}")

fig = go.Figure(go.Indicator
{
        mode = "gauge+number",
        value = probs[1]-probs[0],
        title = {"text": f"Predication:" {userclass}"},
        gauge = {"axis": {"range": [-1,1]},
                 "steps": [
                         {"range": [-1,0], "color":"red"},
                         {"range": [0,1], "color":"green"}],
                 "bar":{"color":"blue"}}
        
        
        
