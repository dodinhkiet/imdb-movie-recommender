import streamlit as st
import pandas as pd
from google.cloud import bigquery
import db_dtypes
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import os
from google.oauth2 import service_account

# Hardcoded configurations
PROJECT_ID = "polynomial-land-477519-s4"
DATASET_ID = "imdb_dataset"

# Page configuration
st.set_page_config(
    page_title="🎬 IMDb Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for better UI
st.markdown("""
<style>
    /* General styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #ff4b4b, #ff7b7b);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    
    .main-header p {
        font-size: 1.2em;
    }
    
    /* Feature sections */
    .feature-section {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }
    
    /* Movie card styling */
    .movie-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 6px solid #ff4b4b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #ff4b4b, #ff7b7b);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
        transition: background 0.3s;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #ff7b7b, #ff4b4b);
    }
    
    /* Metrics */
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
    }
    
    /* Tabs styling */
    .stTabs [data-testid="stTab"] {
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eee;
    }
    
    [data-testid="stSidebar"] .stMetric {
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize BigQuery client
@st.cache_resource
def init_bigquery_client():
    """Khởi tạo BigQuery client từ secrets (dùng trên Streamlit Cloud)"""
    try:
        # Đọc credentials từ secrets
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(
            credentials=credentials,
            project=PROJECT_ID  # vẫn giữ nguyên PROJECT_ID = "polynomial-land-477519-s4"
        )
        
        # Test kết nối
        client.query("SELECT 1").result()
        return client
    except Exception as e:
        st.error(f"Không kết nối được BigQuery: {e}")
        st.info("Kiểm tra lại secrets trong Streamlit Cloud")
        return None

# Load movies with JOIN to get all information (FULL LOAD without vote filter)
@st.cache_data(ttl=3600)
def load_movies_data(_client):
    """Load ALL movies data with JOIN from multiple tables"""
    full_dataset = f"{PROJECT_ID}.{DATASET_ID}"
    
    # Complex JOIN query with CORRECT column names (NO LIMIT, NO min votes filter)
    query = f"""
    WITH MovieGenresAgg AS (
        SELECT 
            mg.movieID,
            STRING_AGG(g.genreName, ', ' ORDER BY g.genreName) as genres
        FROM `{full_dataset}.MovieGenres` mg
        JOIN `{full_dataset}.Genres` g ON mg.genreID = g.genreID
        GROUP BY mg.movieID
    ),
    MovieDirectorsAgg AS (
        SELECT 
            md.movieID,
            STRING_AGG(d.directorName, ', ' ORDER BY d.directorName LIMIT 3) as directors_name
        FROM `{full_dataset}.MovieDirectors` md
        JOIN `{full_dataset}.Directors` d ON md.directorID = d.directorID
        GROUP BY md.movieID
    ),
    MovieWritersAgg AS (
        SELECT 
            mw.movieID,
            STRING_AGG(w.writerName, ', ' ORDER BY w.writerName LIMIT 3) as writers_name
        FROM `{full_dataset}.MovieWriters` mw
        JOIN `{full_dataset}.Writers` w ON mw.writerID = w.writerID
        GROUP BY mw.movieID
    ),
    MovieActorsAgg AS (
        SELECT 
            ma.movieID,
            MAX(CASE WHEN ma.role_order = 1 THEN a.actorName END) as actor_1_name,
            MAX(CASE WHEN ma.role_order = 2 THEN a.actorName END) as actor_2_name,
            MAX(CASE WHEN ma.role_order = 3 THEN a.actorName END) as actor_3_name
        FROM `{full_dataset}.MovieActors` ma
        JOIN `{full_dataset}.Actors` a ON ma.actorID = a.actorID
        WHERE ma.role_order IN (1, 2, 3)
        GROUP BY ma.movieID
    )
    SELECT 
        m.movieID as tconst,
        m.primaryTitle,
        m.startYear,
        m.runtimeMinutes,
        m.averageRating,
        m.numVotes,
        COALESCE(mg.genres, 'Unknown') as genres,
        COALESCE(md.directors_name, 'Unknown') as directors_name,
        COALESCE(mw.writers_name, 'Unknown') as writers_name,
        COALESCE(ma.actor_1_name, '') as actor_1_name,
        COALESCE(ma.actor_2_name, '') as actor_2_name,
        COALESCE(ma.actor_3_name, '') as actor_3_name
    FROM `{full_dataset}.Movies` m
    LEFT JOIN MovieGenresAgg mg ON m.movieID = mg.movieID
    LEFT JOIN MovieDirectorsAgg md ON m.movieID = md.movieID
    LEFT JOIN MovieWritersAgg mw ON m.movieID = mw.movieID
    LEFT JOIN MovieActorsAgg ma ON m.movieID = ma.movieID
    WHERE m.averageRating IS NOT NULL
        AND m.numVotes IS NOT NULL
    ORDER BY m.numVotes DESC
    """
    
    try:
        st.info(f"🔄 Đang load TOÀN BỘ dữ liệu từ {full_dataset}...")
        df = _client.query(query).to_dataframe()
        
        # Handle NaN values
        df['numVotes'] = df['numVotes'].fillna(0)
        df['averageRating'] = df['averageRating'].fillna(0)
        
        st.success(f"✅ Đã tải {len(df):,} phim với đầy đủ thông tin!")
        return df
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        with st.expander("🔍 Xem query"):
            st.code(query, language='sql')
        return pd.DataFrame()

# Load genres list
@st.cache_data(ttl=3600)
def load_genres_data(_client):
    """Load all genres"""
    try:
        query = f"""
        SELECT DISTINCT genreName as genre
        FROM `{PROJECT_ID}.{DATASET_ID}.Genres`
        ORDER BY genreName
        """
        
        df = _client.query(query).to_dataframe()
        genres = df['genre'].tolist()
        st.success(f"✅ Đã tải {len(genres)} thể loại!")
        return genres
    except Exception as e:
        st.warning(f"⚠️ Không thể tải genres: {str(e)}")
        return []

# Content-Based Recommendation Engine (Optimized for memory)
class ContentBasedRecommender:
    def __init__(self, movies_df):
        self.movies_df = movies_df.copy().reset_index(drop=True)  # Ensure index is reset
        self.tfidf_matrix = None
        self._prepare_features()
    
    def _prepare_features(self):
        """Prepare feature vectors for content-based filtering"""
        # Combine features
        self.movies_df['combined_features'] = (
            self.movies_df['genres'].fillna('') + ' ' +
            self.movies_df['directors_name'].fillna('') + ' ' +
            self.movies_df['writers_name'].fillna('') + ' ' +
            self.movies_df['actor_1_name'].fillna('') + ' ' +
            self.movies_df['actor_2_name'].fillna('') + ' ' +
            self.movies_df['actor_3_name'].fillna('')
        )
        
        # TF-IDF vectorization (increase max_features if needed, but monitor memory)
        tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['combined_features'])
    
    def get_recommendations(self, movie_title, n=10):
        """Get movie recommendations based on content similarity (compute on-the-fly to save memory)"""
        # Find movie index
        matching_idx = self.movies_df[self.movies_df['primaryTitle'].str.lower() == movie_title.lower()].index
        
        if len(matching_idx) == 0:
            return pd.DataFrame()
        
        idx = matching_idx[0]  # Take first match if duplicates
        
        # Get the vector for this movie
        movie_vector = self.tfidf_matrix[idx]
        
        # Compute similarity scores (1 x N)
        sim_scores = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        
        # Get top indices
        top_indices = sim_scores.argsort()[::-1][1:n+1]  # Exclude itself and take top n
        
        similarity_scores = sim_scores[top_indices]
        
        recommendations = self.movies_df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarity_scores
        
        return recommendations

# Collaborative Filtering (Rating-based)
class CollaborativeRecommender:
    def __init__(self, movies_df):
        self.movies_df = movies_df.copy()
        self._calculate_weighted_ratings()
    
    def _calculate_weighted_ratings(self):
        """Calculate weighted rating using IMDb formula"""
        C = self.movies_df['averageRating'].mean()
        m = self.movies_df['numVotes'].quantile(0.7) if not self.movies_df['numVotes'].empty else 0
        
        def weighted_rating(row):
            v = row['numVotes']
            R = row['averageRating']
            if m == 0:
                return R
            return (v/(v+m) * R) + (m/(v+m) * C)
        
        self.movies_df['weighted_rating'] = self.movies_df.apply(weighted_rating, axis=1)
    
    def get_top_rated(self, genre=None, min_votes=1000, min_rating=0.0, n=10):
        """Get top rated movies with additional min_rating filter"""
        df = self.movies_df.copy()
        
        # Filter by minimum votes and rating
        df = df[df['numVotes'] >= min_votes]
        df = df[df['averageRating'] >= min_rating]
        
        # Filter by genre if specified
        if genre and genre != "All":
            df = df[df['genres'].str.contains(genre, case=False, na=False)]
        
        df = df.sort_values('weighted_rating', ascending=False)
        
        return df.head(n)

# Hybrid Recommender
class HybridRecommender:
    def __init__(self, content_recommender, collab_recommender):
        self.content_rec = content_recommender
        self.collab_rec = collab_recommender
    
    def get_hybrid_recommendations(self, movie_title, n=10):
        """Combine content-based and collaborative filtering"""
        # Get content-based recommendations
        content_recs = self.content_rec.get_recommendations(movie_title, n*2)
        
        if content_recs.empty:
            return pd.DataFrame()
        
        # Add weighted rating score
        C = self.content_rec.movies_df['averageRating'].mean()
        m = self.content_rec.movies_df['numVotes'].quantile(0.7) if not self.content_rec.movies_df['numVotes'].empty else 0
        
        def weighted_rating(x):
            v = x['numVotes']
            R = x['averageRating']
            if m == 0:
                return R
            return (v/(v+m) * R) + (m/(v+m) * C)
        
        content_recs['weighted_rating'] = content_recs.apply(weighted_rating, axis=1)
        
        # Normalize scores
        content_recs['norm_similarity'] = (content_recs['similarity_score'] - content_recs['similarity_score'].min()) / (content_recs['similarity_score'].max() - content_recs['similarity_score'].min())
        content_recs['norm_rating'] = (content_recs['weighted_rating'] - content_recs['weighted_rating'].min()) / (content_recs['weighted_rating'].max() - content_recs['weighted_rating'].min())
        
        # Hybrid score (70% content, 30% rating)
        content_recs['hybrid_score'] = 0.7 * content_recs['norm_similarity'] + 0.3 * content_recs['norm_rating']
        
        content_recs = content_recs.sort_values('hybrid_score', ascending=False)
        
        return content_recs.head(n)

# Display functions
def display_movie_card(movie, show_similarity=False):
    """Display a movie card"""
    with st.container():
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 🎬 {movie['primaryTitle']}")
            
            # Movie info
            info_parts = []
            if pd.notna(movie['startYear']):
                info_parts.append(f"📅 {int(movie['startYear'])}")
            if pd.notna(movie['runtimeMinutes']):
                info_parts.append(f"⏱️ {int(movie['runtimeMinutes'])} phút")
            if movie['genres'] and movie['genres'] != 'Unknown':
                info_parts.append(f"🎭 {movie['genres']}")
            
            st.markdown(" | ".join(info_parts))
            
            # Cast and crew
            if movie['directors_name'] and movie['directors_name'] != 'Unknown':
                st.markdown(f"**🎬 Đạo diễn:** {movie['directors_name']}")
            
            actors = [a for a in [movie.get('actor_1_name'), movie.get('actor_2_name'), movie.get('actor_3_name')] 
                     if pd.notna(a) and a != '']
            if actors:
                st.markdown(f"**👥 Diễn viên:** {', '.join(actors)}")
        
        with col2:
            # Rating
            if pd.notna(movie['averageRating']):
                st.metric("⭐ Rating", f"{movie['averageRating']:.1f}/10")
            if pd.notna(movie['numVotes']):
                st.metric("👍 Votes", f"{int(movie['numVotes']):,}")
            
            # Similarity score
            if show_similarity and 'similarity_score' in movie:
                st.metric("🎯 Tương đồng", f"{movie['similarity_score']:.0%}")
            elif show_similarity and 'hybrid_score' in movie:
                st.metric("🎯 Điểm đề xuất", f"{movie['hybrid_score']:.0%}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<div class="main-header"><h1>🎬 Hệ thống Đề xuất Phim IMDb</h1><p>Tìm kiếm, khám phá và nhận gợi ý phim hay từ cơ sở dữ liệu IMDb</p></div>', unsafe_allow_html=True)
    
    # Sidebar for stats only
    with st.sidebar:
        st.markdown("### 📊 Thống kê Tổng quan")
    
    # Initialize client directly
    if 'client' not in st.session_state:
        st.session_state.client = init_bigquery_client()
    
    client = st.session_state.client
    
    if client is None:
        st.error("❌ Không thể kết nối đến BigQuery. Vui lòng kiểm tra credentials.")
        return
    
    # Load data automatically
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Đang JOIN và tải TOÀN BỘ dữ liệu từ 9 bảng... (Có thể mất 2-5 phút)"):
            movies_df = load_movies_data(client)
            genres_list = load_genres_data(client)
            
            if not movies_df.empty:
                st.session_state.movies_df = movies_df
                st.session_state.genres_list = genres_list
                
                # Initialize recommenders
                with st.spinner("🤖 Đang khởi tạo recommendation engines... (Có thể mất vài phút cho dataset lớn)"):
                    st.session_state.content_rec = ContentBasedRecommender(movies_df)
                    st.session_state.collab_rec = CollaborativeRecommender(movies_df)
                    st.session_state.hybrid_rec = HybridRecommender(
                        st.session_state.content_rec,
                        st.session_state.collab_rec
                    )
                
                st.session_state.data_loaded = True
                
                st.balloons()
                st.success("🎉 TOÀN BỘ dữ liệu đã được tải thành công!")
                st.info(f"📊 Load {len(movies_df):,} phim với đầy đủ genres, directors, actors!")
            else:
                st.error("❌ Không thể tải dữ liệu!")
                return
    
    movies_df = st.session_state.movies_df
    genres_list = st.session_state.genres_list
    
    # Sidebar stats (updated for full dataset)
    with st.sidebar:
        st.metric("🎬 Tổng số phim", f"{len(movies_df):,}")
        st.metric("🎭 Số thể loại", len(genres_list))
        avg_rating = movies_df['averageRating'].mean()
        st.metric("⭐ Rating TB", f"{avg_rating:.1f}/10")
        median_votes = movies_df['numVotes'].median()
        st.metric("📊 Votes trung vị", f"{int(median_votes):,}")
    
    # Use tabs for different modes to make UI nicer
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Tìm kiếm phim", "🎯 Phim tương tự", "🏆 Top phim", "🎭 Theo thể loại", "📊 Thống kê"])
    
    with tab1:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.header("🔍 Tìm kiếm phim")
        
        # Search box with autocomplete suggestion
        search_query = st.text_input("Nhập tên phim:", placeholder="Ví dụ: The Godfather, Inception...")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            year_min = st.number_input("Năm từ", min_value=1900, max_value=2024, value=1990)
        with col2:
            year_max = st.number_input("Năm đến", min_value=1900, max_value=2024, value=2024)
        with col3:
            min_rating = st.slider("Rating tối thiểu", 0.0, 10.0, 6.0, 0.1)
        with col4:
            min_votes = st.slider("Votes tối thiểu", 0, 100000, 1000, 100)
        
        search_type = st.radio("Loại tìm kiếm:", ["Chứa từ khóa", "Bắt đầu bằng từ khóa"])
        
        if search_query:
            # Filter movies
            if search_type == "Bắt đầu bằng từ khóa":
                filtered_df = movies_df[
                    movies_df['primaryTitle'].str.lower().str.startswith(search_query.lower(), na=False) &
                    (movies_df['startYear'] >= year_min) &
                    (movies_df['startYear'] <= year_max) &
                    (movies_df['averageRating'] >= min_rating) &
                    (movies_df['numVotes'] >= min_votes)
                ].sort_values('averageRating', ascending=False)
            else:
                filtered_df = movies_df[
                    movies_df['primaryTitle'].str.contains(search_query, case=False, na=False) &
                    (movies_df['startYear'] >= year_min) &
                    (movies_df['startYear'] <= year_max) &
                    (movies_df['averageRating'] >= min_rating) &
                    (movies_df['numVotes'] >= min_votes)
                ].sort_values('averageRating', ascending=False)
            
            st.subheader(f"Tìm thấy {len(filtered_df)} kết quả")
            
            for idx, movie in filtered_df.head(20).iterrows():
                display_movie_card(movie)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.header("🎯 Tìm phim tương tự")
        
        # Movie selection
        movie_titles = sorted(movies_df['primaryTitle'].unique())
        selected_movie = st.selectbox("Chọn một bộ phim:", movie_titles)
        
        col1, col2 = st.columns(2)
        with col1:
            rec_method = st.radio("Phương pháp đề xuất:", 
                                 ["Content-Based", "Hybrid (Recommended)"])
        with col2:
            n_recommendations = st.slider("Số lượng đề xuất:", 5, 20, 10)
        
        if st.button("🎯 Tìm phim tương tự", type="primary"):
            with st.spinner("Đang phân tích và tìm kiếm..."):
                if rec_method == "Content-Based":
                    recommendations = st.session_state.content_rec.get_recommendations(
                        selected_movie, n_recommendations
                    )
                else:
                    recommendations = st.session_state.hybrid_rec.get_hybrid_recommendations(
                        selected_movie, n_recommendations
                    )
                
                if not recommendations.empty:
                    st.success(f"✅ Tìm thấy {len(recommendations)} phim tương tự với **{selected_movie}**")
                    
                    for idx, movie in recommendations.iterrows():
                        display_movie_card(movie, show_similarity=True)
                else:
                    st.warning("Không tìm thấy phim tương tự.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.header("🏆 Top phim được đánh giá cao")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_votes = st.number_input("Số vote tối thiểu:", 0, 100000, 5000, 1000)
        with col2:
            min_rating = st.slider("Rating tối thiểu", 0.0, 10.0, 7.0, 0.1)
        with col3:
            n_top = st.slider("Hiển thị top:", 5, 50, 20)
        
        top_movies = st.session_state.collab_rec.get_top_rated(
            genre=None, min_votes=min_votes, min_rating=min_rating, n=n_top
        )
        
        st.subheader(f"Top {len(top_movies)} phim")
        
        for idx, movie in top_movies.iterrows():
            display_movie_card(movie)
        
        # Visualization
        st.subheader("📊 Phân bố Rating")
        fig = px.histogram(top_movies, x='averageRating', nbins=20,
                          title="Phân bố điểm rating",
                          labels={'averageRating': 'Rating', 'count': 'Số lượng'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.header("🎭 Top phim theo thể loại")
        
        selected_genre = st.selectbox("Chọn thể loại:", ["All"] + genres_list)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_votes = st.number_input("Số vote tối thiểu:", 0, 100000, 1000, 500)
        with col2:
            min_rating = st.slider("Rating tối thiểu", 0.0, 10.0, 6.5, 0.1)
        with col3:
            n_top = st.slider("Số lượng phim:", 5, 30, 15)
        
        top_movies = st.session_state.collab_rec.get_top_rated(
            genre=selected_genre, min_votes=min_votes, min_rating=min_rating, n=n_top
        )
        
        st.subheader(f"Top {len(top_movies)} phim {selected_genre}")
        
        for idx, movie in top_movies.iterrows():
            display_movie_card(movie)
        
        # Genre statistics
        if selected_genre != "All":
            genre_df = movies_df[
                (movies_df['genres'].str.contains(selected_genre, case=False, na=False)) &
                (movies_df['numVotes'] >= min_votes) &
                (movies_df['averageRating'] >= min_rating)
            ]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Tổng số phim", f"{len(genre_df):,}")
            with col2:
                st.metric("⭐ Rating TB", f"{genre_df['averageRating'].mean():.1f}" if not genre_df.empty else "N/A")
            with col3:
                st.metric("⏱️ Độ dài TB", f"{int(genre_df['runtimeMinutes'].mean())} phút" if not genre_df.empty else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.header("📊 Thống kê tổng quan (TOÀN BỘ dữ liệu)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎬 Tổng phim", f"{len(movies_df):,}")
        with col2:
            st.metric("⭐ Rating TB", f"{movies_df['averageRating'].mean():.2f}")
        with col3:
            st.metric("⏱️ Độ dài TB", f"{int(movies_df['runtimeMinutes'].mean())} phút" if not movies_df['runtimeMinutes'].isnull().all() else "N/A")
        with col4:
            st.metric("📊 Votes TB", f"{int(movies_df['numVotes'].mean()):,}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Phân bố Rating")
            fig = px.histogram(movies_df, x='averageRating', nbins=30,
                              title="Phân bố điểm Rating (Toàn bộ dataset)",
                              color_discrete_sequence=['#FF4B4B'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎭 Top 10 thể loại")
            all_genres = []
            for genres in movies_df['genres'].dropna():
                if genres != 'Unknown':
                    all_genres.extend([g.strip() for g in str(genres).split(',')])
            
            genre_counts = pd.Series(all_genres).value_counts().head(10)
            fig = px.bar(x=genre_counts.values, y=genre_counts.index, 
                        orientation='h', title="Thể loại phổ biến (Toàn bộ)",
                        color=genre_counts.values,
                        color_continuous_scale='Reds')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()