from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('flipkart_com-ecommerce_sample.csv')

# Removing missing value
data['retail_price'].fillna(data['retail_price'].mean(), inplace=True)
data['discounted_price'].fillna(data['discounted_price'].mean(), inplace=True)
data['image'].fillna("No image", inplace=True)
data['is_FK_Advantage_product'].fillna("No is_FK_Advantage_product", inplace=True)
data['uniq_id'].fillna("No uniq_id", inplace=True)
data['crawl_timestamp'].fillna(data['crawl_timestamp'].mode()[0], inplace=True)
data['product_url'].fillna("", inplace=True)
data['product_name'].fillna("", inplace=True)
data['product_category_tree'].fillna(data['product_category_tree'].mode()[0], inplace=True)
data['pid'].fillna("No pid", inplace=True)
data['brand'].fillna(data['brand'].mode()[0], inplace=True)
data['description'].fillna("No description", inplace=True)
data['product_rating'].fillna("No product_rating", inplace=True)
data['overall_rating'].fillna("No overall_rating", inplace=True)
data['product_specifications'].fillna("No product_specifications", inplace=True)

# Vectorize product names
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['product_name'].fillna(''))

# Function to retrieve relevant products
def retrieve_products(query, top_n=20):
    query_vec = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return data.iloc[top_indices]

# Streamlit app
st.title("Product Search System")

query = st.text_input("Enter search term:")
if query:
    results = retrieve_products(query)
    st.write("Top results:")
    for i, row in results.iterrows():
        st.write(f"**{row['product_name']}**")
        st.write(f"URL: {row['product_url']}")
        st.write(f"Retail Price: {row['retail_price']}, Discounted Price: {row['discounted_price']}")
        st.write("---")
