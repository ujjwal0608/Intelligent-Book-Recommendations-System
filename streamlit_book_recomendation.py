import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Set the page config with improved aesthetics
st.set_page_config(page_title="📚 Book Recommender", layout="wide")

@st.cache_data
def load_data():
    
    return pd.read_csv(r"/Users/ujjwalraj/Desktop/Book-Recomendation-System/Cluster_cleaned.csv")

@st.cache_resource
def load_models():
    
    tfidf = joblib.load(r"/Users/ujjwalraj/Desktop/Book-Recomendation-System/tfidf.pkl")
    cosine_sim = joblib.load(r"/Users/ujjwalraj/Desktop/Book-Recomendation-System/cosine_sim_matrix.pkl")
    return tfidf, cosine_sim

df = load_data()
tfidf, cosine_sim = load_models()

book_indices = pd.Series(df.index, index=df['Book Name']).drop_duplicates()
cluster_map = df.set_index('Book Name')['cluster'].to_dict()

# ---------------- Recommendation Functions ---------------- #

def get_content_based_recommendations(title, top_n=5):
    
    if title not in book_indices:
        return pd.DataFrame()
    
    idx = book_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices_similar = [i[0] for i in sim_scores]
    return df.iloc[book_indices_similar][['Book Name', 'Author', 'Genre', 'Rating']]

def get_clustering_based_recommendations(preference_type, preference_input, top_n=5):
    
    if preference_type == "Genre":
        cluster_books = df[df['Genre'] == preference_input]
        
    else:
        cluster_books = df[df['Author'] == preference_input]

    return cluster_books.sort_values(by='Rating', ascending=False).head(top_n)

def get_hybrid_recommendations(preference_type, preference_input, top_n=5):
    
    filtered = df[df[preference_type] == preference_input]
    return filtered.sort_values(by='Rating', ascending=False).head(top_n)

def plot_wordcloud(text, title="Word Cloud"):
    
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        colormap='viridis',
        width=800,
        height=400
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# ---------------- Sidebar Navigation ----------------
page = st.sidebar.selectbox("Navigate", ["🏠 Home", "📊 EDA", "🔍 Recommendation System"])

# ---------------- HOME ----------------
if page == "🏠 Home":
    
    st.markdown("<h1 style='text-align: center; color: #4A4A4A;'>📚 Welcome to Your Personal Book Recommender</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 18px; padding: 10px 0; color: #5E5E5E;'>
    Discover books tailored to your taste and interests using smart algorithms.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ✅ What You Can Do:")
    st.markdown("""
    - 🔍 **Explore books** by genre, author, or book title.
    - 📈 **Visualize trends** with interactive EDA.
    - 🤖 **Get smart recommendations** using content, clustering & hybrid models.
    """)
    st.markdown("---")

# ---------------- EDA ----------------
elif page == "📊 EDA":
    
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("---")

    st.markdown("## 🟢 Easy Level Insights")

    # Most popular genres
    genre_counts = df['Genre'].value_counts().head(10).reset_index()
    genre_counts.columns = ['Genre', 'Count']
    st.plotly_chart(px.bar(genre_counts, x='Genre', y='Count', title='Top 10 Most Popular Genres'), use_container_width=True)

    # Top authors by rating
    top_authors = df.groupby('Author')['Rating'].mean().sort_values(ascending=False).head(10).reset_index()
    st.plotly_chart(px.bar(top_authors, x='Author', y='Rating', title='Top 10 Authors by Average Book Rating'), use_container_width=True)

    # Rating distribution
    st.plotly_chart(px.histogram(df, x='Rating', nbins=20, title='Average Rating Distribution'), use_container_width=True)

    # Reviews vs Ratings
    st.plotly_chart(px.scatter(df, x='Number of Reviews', y='Rating', 
                                title='Ratings vs Number of Reviews', trendline='ols'), use_container_width=True)

    st.markdown("---")
    st.markdown("## 🟡 Medium Level Insights")

    # Cluster examples
    cluster_books = df.groupby('cluster')['Book Name'].apply(lambda x: ', '.join(x.head(3))).reset_index()
    st.write("📚 **Top Books in Each Cluster:**")
    st.dataframe(cluster_books.rename(columns={'Book Name': 'Top Books in Cluster'}))

    genre_rating = df.groupby('Genre')['Rating'].mean().sort_values(ascending=False).head(10).reset_index()
    st.plotly_chart(px.bar(genre_rating, x='Genre', y='Rating', title='Genre Similarity and Average Ratings'), use_container_width=True)

    st.markdown("---")
    st.markdown("## 🔴 Scenario-Based Insights")

    sci_fi_recs = df[df['Genre'].str.contains("science fiction", case=False)].sort_values(by='Rating', ascending=False).head(5)
    st.write("👽 **Top 5 Science Fiction Books:**")
    st.dataframe(sci_fi_recs[['Book Name', 'Author', 'Rating']])

    hidden_gems = df[df['Number of Reviews'] < 50].sort_values(by='Rating', ascending=False).head(5)
    st.write("💎 **Hidden Gems (High Rating, Low Reviews):**")
    st.dataframe(hidden_gems[['Book Name', 'Author', 'Rating', 'Number of Reviews']])

    st.markdown("### 🧠 Word Cloud Generator")
    genre_choice = st.selectbox("Select Genre for Word Cloud", df['Genre'].dropna().unique())
    if st.button("Generate Word Cloud"):
        genre_text = " ".join(df[df['Genre'] == genre_choice]['cleaned_text'].dropna().astype(str))
        plot_wordcloud(genre_text, f"{genre_choice} Word Cloud")

# ---------------- Recommendation System ----------------
elif page == "🔍 Recommendation System":
    st.title("🔍 Personalized Book Recommendation System")
    st.markdown("---")

    st.subheader("🎯 Choose Your Preferences")
    preference_type = st.radio("Choose a Preference Filter", ["Genre", "Author", "Book Name"])
    
    if preference_type == "Genre":
        preference_input = st.selectbox("Select a Genre", sorted(df['Genre'].dropna().unique()))
    elif preference_type == "Author":
        preference_input = st.selectbox("Select an Author", sorted(df['Author'].dropna().unique()))
    else:
        preference_input = st.selectbox("Select a Book", sorted(df['Book Name'].dropna().unique()))

    top_n = st.slider("Number of Recommendations", min_value=3, max_value=10, value=5)

    st.subheader("🔢 Choose Recommendation Method")
    rec_type = st.radio("Select Recommendation Method", ["Clustering-Based", "Content-Based", "Hybrid"])

    if st.button("📚 Get Recommendations"):
        with st.spinner("Finding the perfect reads..."):
            if rec_type == "Clustering-Based":
                result = get_clustering_based_recommendations(preference_type, preference_input, top_n)
            elif rec_type == "Content-Based":
                result = get_content_based_recommendations(preference_input, top_n)
            else:
                result = get_hybrid_recommendations(preference_type, preference_input, top_n)

        if not result.empty:
            st.success("✅ Here are your recommended books:")

            for i in range(0, len(result), 2):
                cols = st.columns(2)
                for col_index in range(2):
                    if i + col_index < len(result):
                        book = result.iloc[i + col_index]
                        with cols[col_index]:
                            st.markdown(f"""
    <div style='
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        height: auto;
        color: #333333;
        font-family: sans-serif;
    '>
        <h4 style='color:#1a1a1a;'>{book['Book Name']}</h4>
        <p><strong>Author:</strong> {book['Author']}</p>
        <p><strong>Genre:</strong> {book.get('Genre', 'N/A')}</p>
        <p><strong>Rating:</strong> ⭐ {book['Rating']}</p>
    </div>
""", unsafe_allow_html=True)


            st.metric("📊 Average Rating", round(result['Rating'].mean(), 2))
            st.metric("📚 Total Recommendations", len(result))

            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Recommendations", csv, "recommendations.csv", "text/csv")
        else:
            st.warning("⚠️ No recommendations found based on your input.")
