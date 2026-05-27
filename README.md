# 📚 Personalized Book Recommendation System

An interactive, data-driven web application that provides customized book recommendations using advanced Natural Language Processing (NLP) and machine learning algorithms. Built natively using Streamlit, this platform features an interactive data visualization dashboard and a deployment panel powered by a tripartite recommendation framework.

---

## 🚀 Features

* **Tripartite Recommendation Engine:** Explores books through three distinct lenses to give highly accurate suggestions.
* **Interactive Dashboard:** Beautiful data visualizations built with Streamlit to explore book trends, ratings, and genres.
* **User-Friendly UI:** Clean, minimalist web interface designed for seamless book discovery.
* **Fast Inferences:** Optimized backend computations for real-time recommendation updates.

---

## 🛠️ The Tripartite Framework

The core engine utilizes three primary methodologies to recommend books:

1.  **Popularity-Based Filtering:** Recommends trending and top-rated books globally based on collective user ratings (ideal for new users/cold start).
2.  **Collaborative Filtering:** Analyzes user behavior, patterns, and similarities to suggest books that similar readers enjoyed.
3.  **Content-Based Filtering (NLP):** Utilizes Natural Language Processing to evaluate book metadata (genres, authors, descriptions) and find titles structurally similar to your favorites.

---

## 📊 Dataset

The system is trained and evaluated on a comprehensive book dataset containing:
* **Books Metadata:** Title, Author, Year of Publication, Publisher, and Image URLs.
* **User Data:** Anonymous user IDs and demographic locations.
* **Ratings Data:** Explicit and implicit book ratings recorded on a scale.

---

## 💻 Tech Stack

* **Frontend & Dashboard:** Streamlit
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & NLP:** Scikit-Learn (Cosine Similarity, Nearest Neighbors)
* **Visualizations:** Plotly / Matplotlib / Seaborn

---

## ⚙️ Installation & Setup

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/ujjwal0608/Intelligent-Book-Recommendations-System.git](https://github.com/ujjwal0608/Intelligent-Book-Recommendations-System.git)
cd Intelligent-Book-Recommendations-System
