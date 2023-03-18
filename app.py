import streamlit as st
import pandas as pd
import numpy as np
from prediction import xgb_predict, lr_predict
import nltk
from nltk import sent_tokenize
nltk.download('punkt');

#st.write(st.__version__)

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #400745;
    }
    div.stButton > button:first-child {
        background-color: #ffffff;
        color:#000000;
    }
    .css-z09lfk:focus:not(:active) {
        border-color: #ffffff;
        box-shadow: none;
        background-color: #ffffff;
        color:#000000;
    }
    .css-z09lfk:focus:(:active) {
        border-color: #ffffff;
        box-shadow: #832196;
        color: #ffffff;
        background-color: #000000;
    }
    .css-z09lfk:focus:(:active){
        background-color: #000000;
        border-color: #ffffff;
        box-shadow: #832196;
        color: #ffffff;
        background-color: #000000;
    }

</style>
""", unsafe_allow_html=True)

def RunAI(string, classifier):
    sentences = sent_tokenize(string)
    flags = []
    if classifier == "XGBoost":
      for sentence in sentences:
        if xgb_predict(sentence) == 1:
          flags.append(sentence)
      return flags

    elif classifier == "LogisticRegression":
      for sentence in sentences:
        if lr_predict(sentence) == 1:
          flags.append(sentence)
      return flags
    

def Display(essay, col1, col2, col3):
    if essay != "":
        st.info("Loading...")
        xgflags = RunAI(essay, "XGBoost")
        lrflags = RunAI(essay, "LogisticRegression")
        count = 0
        
        with col1:
            for flag in lrflags:
                count+=1
                st.write(str(count)+". "+flag)
            if count == 0:
                st.write("No red flags found!")

        count = 0
        
        with col2:
            for flag in xgflags:
                count += 1
                st.write(str(count)+". "+flag)
            if count == 0:
                st.write("No red flags found!")

        st.success("Calculations complete!")

st.title(":red[under construction lol]")
st.title("Welcome to :violet[T.E.D.D.Y]")
st.subheader("Text-based Early Depression Detector for Youth", anchor="welcome-to-t-e-d-d-y")
st.text("Enter your essay into the sidebar.")
st.text("")

col1, col2, col3 = st.columns(3)
with col1:
    col1h = st.header("High-Sensitivity Test")
    st.caption("_with Logistic Regression_")

with col2:
    col2h = st.header("Low-Sensitivity Test")
    st.caption("_with XGBoost_")

with col3:
    st.header("Grammatical Test")

with st.sidebar:
    st.title("Copy-paste an essay into the box below!")
    essay = st.text_input("Enter text here...")
    st.button("Submit", on_click=Display(essay, col1, col2, col3))
