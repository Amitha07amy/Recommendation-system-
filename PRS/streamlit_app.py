import streamlit as st

st.set_page_config(page_title="Product Recommendation", layout="wide")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import html

@st.cache_data
def load_data():
    return pd.read_csv("sample-data.csv")

df = load_data()

df["description"] = df["description"].apply(lambda x: html.unescape(str(x)))

@st.cache_data
def load_data():
    return pd.read_csv("sample-data.csv")

df = load_data()

df["description"] = df["description"].apply(lambda x: html.unescape(str(x)))

def get_short_description(text):
    sentences = text.split(". ")[:2] 
    return ". ".join(sentences) + "."

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"])
similarity_matrix = cosine_similarity(tfidf_matrix)


st.title("🔍 Product Recommendation System")
st.write("By Amitha")
st.markdown("### **Find products similar to your selection!**")

st.sidebar.header("🔽 Select a Product")
selected_id = st.sidebar.selectbox("Choose a product ID:", df["id"].tolist())

st.subheader("🛍 Selected Product Details")
selected_product = df[df["id"] == selected_id].iloc[0]
short_description = get_short_description(selected_product["description"])
st.markdown(f"**Product ID:** {selected_product['id']}")
st.markdown(f"*{short_description}*")

if st.button("📜 Expand Description"):
    st.markdown(f"*{selected_product['description']}*")

idx = df.index[df["id"] == selected_id][0]
similarities = list(enumerate(similarity_matrix[idx]))
similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[1:4]

st.markdown("### 🔗 **Related Products**")
col1, col2, col3 = st.columns(3)

for i, (col, (sim_idx, score)) in enumerate(zip([col1, col2, col3], similarities)):
    related_product = df.iloc[sim_idx]
    related_short_description = get_short_description(related_product["description"])
    
    with col:
        st.markdown(f"**🔹 {related_product['id']}**")
        st.markdown(f"*{related_short_description}*")

