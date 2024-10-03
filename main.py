import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import glob
from pathlib import Path

filepaths = glob.glob("diary/*.txt")
analyzer = SentimentIntensityAnalyzer()
scores = []
dates = []
for filepath in filepaths:
    with open(filepath) as file:
        text = file.read()
        score = analyzer.polarity_scores(text)
        scores.append(score)
        date = Path(filepath).stem
        dates.append(date)

st.title("Diary Tone")

st.header("Positivity")
pos_data = [dict["pos"] for dict in scores]
pos_figure = px.line(x=dates, y=pos_data,
                     labels={'x': "dates", 'y': 'Positivity'})
st.plotly_chart(pos_figure)

st.header("Negativity")
neg_data = [dict["neg"] for dict in scores]
neg_figure = px.line(x=dates, y=neg_data,
                     labels={'x': "dates", 'y': 'Negativity'})
st.plotly_chart(neg_figure)
