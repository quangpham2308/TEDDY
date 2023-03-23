import streamlit as st
import pandas as pd
import numpy as np
from prediction import xgb_predict, lr_predict, xgb_suicide, lr_suicide
import nltk
from nltk import sent_tokenize, word_tokenize
nltk.download('punkt');

#st.write(st.__version__)

def RunAI(string, classifier):
    sentences = sent_tokenize(string)

    flags = []
    danger_words = ["die", "kill", "death", "gun", "bomb", "jump", "cut", "hurt", "harm", "threat", "abuse", "abused", "violence", "hopeless", "despair", "emptiness", "suicide", "shoot", "stab", "sad", "depress", "depressed", "depression"]
    
    if classifier == "XGBoost":
      for sentence in sentences:
        if xgb_predict(sentence) == 1:
          flags.append(sentence)

    elif classifier == "LogisticRegression":
      for sentence in sentences:
        if lr_predict(sentence) == 1:
          flags.append(sentence)

    elif classifier == "XGBSuicide":
      for sentence in sentences:
        if xgb_suicide(sentence) == 1:
          flags.append(sentence)
    
    elif classifier == "LogisticSuicide":
      for sentence in sentences:
        if lr_suicide(sentence) == 1:
          flags.append(sentence)

    elif classifier == "WordTest":
        for sentence in sentences:
            words = word_tokenize(sentence)
            for word in words:
                if word in danger_words:
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
        wrflags = RunAI(essay, "WordTest")

        with col1:
            ShowFlags(wrflags)
            
        with col2:
            ShowFlags(lrflags)
        
        with col3:
            ShowFlags(xgflags)
        
        with col4:
            ShowFlags(xgsflags)

        st.success("Calculations complete!")

st.title("Welcome to :violet[T.E.D.D.Y.]")
st.subheader("Text-based Early Distress Detector for Youth", anchor="welcome-to-t-e-d-d-y")
st.caption("_Use the sidebar to enter an essay or chat conversation! T.E.D.D.Y. will use artificial intelligence to display sentences that may be a cause for concern.\n :red[T.E.D.D.Y. is not meant to be used as a diagnostic tool] - it is designed to give you a general idea of whether someone in your school or workplace might need more emotional support._")
st.write("\n")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Concerning Words Test")
    st.caption("_The following sentences include words that may be a cause for concern._")
    
with col2:
    st.subheader("Low-Risk Test")
    st.caption("_The writer may be experiencing some struggles that need to be checked._")

with col3:
    st.subheader("Mid-Risk Test")
    st.caption("_The writer might be expressing depressive thoughts._")
    
with col4:
    st.subheader("High-Risk Test")
    st.caption("_The writer could be struggling with self-destructive thoughts._")

with st.sidebar:
    st.title("Copy-paste an essay / conversation into the box below!")
    essay = st.text_input("Enter text here...")
    submit = st.button("Submit", on_click=Display(essay))
    st.write("[Help us improve T.E.D.D.Y!](https://forms.gle/eAYpKmd9udkdFUir6)")
