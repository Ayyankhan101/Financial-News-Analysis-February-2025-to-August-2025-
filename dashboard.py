# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# ------------------------------------------------------------------
# CONFIG & STYLE
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Financial News Dashboard",
    page_icon="📈",
    layout="wide",
)

# ------------------------------------------------------------------
# DATA LOADING & CLEANING
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("financial_news_events.csv", parse_dates=["Date"])
    df = df[df["Headline"].notna()].reset_index(drop=True)
    num_cols = ["Index_Change_Percent", "Trading_Volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Sentiment"] = df["Sentiment"].fillna("Unknown")
    return df

df = load_data()

# helper for logistic model
@st.cache_data
def train_model(data):
    df_ = data.copy()
    cat_cols = ["Sentiment", "Sector", "Impact_Level", "Market_Event"]
    for c in cat_cols:
        df_[c+"_enc"] = LabelEncoder().fit_transform(df_[c].astype(str))
    df_["Direction"] = (df_["Index_Change_Percent"] > 0).astype(int)
    X = df_[[c+"_enc" for c in cat_cols]].dropna()
    y = df_.loc[X.index, "Direction"]
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)
    report = classification_report(y, model.predict(X), output_dict=True)
    return pd.DataFrame(report).transpose()

report_df = train_model(df)

# ------------------------------------------------------------------
# PAGE HEADER
# ------------------------------------------------------------------
st.title("📊 Financial News Events – Executive Dashboard")
st.markdown("A quick look at **1 000+ market-moving headlines** and their impact.")

# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Events", len(df))
kpi2.metric("Unique Sectors", df["Sector"].nunique())
kpi3.metric("Avg Index Change", f"{df['Index_Change_Percent'].mean():.2f}%")
kpi4.metric("Up-Market Share", f"{(df['Index_Change_Percent']>0).mean():.1%}")



# ------------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------------
st.subheader("📈 Quick Model: Sentiment + Metadata → Up/Down")
st.write("Logistic Regression with balanced class weights.")
st.dataframe(report_df.style.highlight_max(axis=0))

# ------------------------------------------------------------------
# LAYOUT: 2 COLS
# ------------------------------------------------------------------

st.subheader("🗣️ Sentiment vs Index Change")
fig, ax = plt.subplots(figsize=(8,3))
sns.boxplot(x="Sentiment", y="Index_Change_Percent", data=df, palette="viridis", ax=ax)
ax.set_title("Index Change % by Sentiment")
st.pyplot(fig)

st.subheader("🏭 Sector × Impact Level")
pivot = pd.crosstab(df["Sector"], df["Impact_Level"])
fig, ax = plt.subplots(figsize=(7,4))
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
ax.set_title("Event Counts")
st.pyplot(fig)

st.subheader("🔤 Word Cloud – Headlines")
text = " ".join(df["Headline"].dropna())
wc = WordCloud(width=600, height=300, background_color="white",
                   colormap="tab10").generate(text)
fig, ax = plt.subplots(figsize=(7,3))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
