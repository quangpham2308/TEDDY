import streamlit as st
import pandas as pd
import numpy as np
from prediction import xgb_predict, lr_predict, xgb_suicide, lr_suicide
import nltk
from nltk import sent_tokenize
nltk.download('punkt');

#st.write(st.__version__)

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

    elif classifier == "XGBSuicide":
      for sentence in sentences:
        if xgb_suicide(sentence) == 1:
          flags.append(sentence)
      return flags
    
    elif classifier == "LogisticSuicide":
      for sentence in sentences:
        if lr_suicide(sentence) == 1:
          flags.append(sentence)
      return flags
    

def ShowFlags(flaglist):
    count = 0
    flagcount = len(flaglist)
    if flagcount == 0:
        st.write("No red flags found!")
    else:
        with st.expander("Found "+str(flagcount)+" red flag/s:", True):
            for flag in flaglist:
                count+=1
                st.write(str(count)+". "+flag)
            st.write("\n")

def Display(essay):
    if essay != "":
        st.info("Loading...")
        
        xgflags = RunAI(essay, "XGBoost")
        lrflags = RunAI(essay, "LogisticRegression")
        xgsflags = RunAI(essay, "XGBSuicide")
        lrsflags = RunAI(essay, "LogisticSuicide")
        
        with col1:
            ShowFlags(lrflags)
        
        with col2:
            ShowFlags(xgflags)
        
        with col3:
            ShowFlags(lrsflags)
        
        with col4:
            ShowFlags(xgsflags)

        st.success("Calculations complete!")

st.title("Welcome to :violet[T.E.D.D.Y.]")
st.subheader("Text-based Early Distress Detector for Youth", anchor="welcome-to-t-e-d-d-y")
st.write("Enter your essay into the sidebar.")
st.write("\n")

col1, col2, col3 = st.columns(3)

with col1:
    col1h = st.subheader("Low-Risk Test")

with col2:
    col2h = st.subheader("Mid-Risk Test")
    
with col3:
    st.subheader("High-Risk Test")

with st.sidebar:
    st.title("Copy-paste an essay into the box below!")
    essay = st.text_input("Enter text here...")
    submit = st.button("Submit", on_click=Display(essay))
