import streamlit as st
import pandas as pd
import random # For generating random colors

from recommendation_system import (
    load_sample_data,
    calculate_content_similarity,
    create_user_item_matrix,
    calculate_collaborative_similarity,
    calculate_track_popularity,
    hybrid_recommendation
)

# --- Load Data and Precompute Similarities (Cached for performance) ---
@st.cache_data
def load_and_preprocess_data():
    """Loads sample data and precomputes all necessary matrices."""
    st.write("Loading and preprocessing sample data... This might take a moment.")
    lastfm_df, merged_df = load_sample_data()

    if lastfm_df.empty or merged_df.empty:
        st.error("Failed to load sample data. Please ensure 'last.fm_data_sample.csv' and 'tracks_sample.csv' are in the same directory and have the correct column names (Track, Artist, Username, Genre).")
        st.stop()

    content_similarity_matrix, merged_df_with_features = calculate_content_similarity(merged_df)
    user_item_matrix = create_user_item_matrix(lastfm_df)
    user_similarity_matrix = calculate_collaborative_similarity(user_item_matrix)
    track_popularity = calculate_track_popularity(lastfm_df)

    st.success("Sample data loaded and preprocessed!")
    return lastfm_df, merged_df_with_features, user_item_matrix, user_similarity_matrix, content_similarity_matrix, track_popularity

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Hybrid Music Recommender")

st.title("🎶 Hybrid Music Recommendation System (Sample Data)")
st.markdown("""
Welcome to the Hybrid Music Recommender! This system combines content-based and collaborative filtering
to suggest music tailored to your taste, with an option to penalize popular tracks for more diverse recommendations.
**Note: This application is running on a sample dataset.** For a full-scale application,
more robust data handling and potentially more efficient algorithms would be required.
""")

# Load data and precompute when the app starts
lastfm_df, merged_df_with_features, user_item_matrix, user_similarity_matrix, content_similarity_matrix, track_popularity = load_and_preprocess_data()

# Initialize session state for recommendations and current song index
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()
if 'current_song_index' not in st.session_state:
    st.session_state.current_song_index = 0

# Get available user IDs from the loaded sample data
available_user_ids = user_item_matrix.index.tolist()
if 'NewUser123' not in available_user_ids:
    available_user_ids.append('NewUser123') # Add a dummy new user for cold start demo

# Sidebar for user inputs
st.sidebar.header("User Input & Settings")

user_id_input = st.sidebar.selectbox(
    "Select or Enter User ID",
    options=available_user_ids,
    index=available_user_ids.index('Babs_05') if 'Babs_05' in available_user_ids else 0, # Default to Babs_05 if available, else first user
    help="Choose an existing user from the sample data or enter a new ID for a cold start demo."
)

num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=20,
    value=10
)

st.sidebar.subheader("Recommendation Weights")
content_weight = st.sidebar.slider(
    "Content-Based Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Higher value means more recommendations based on song attributes (genre, artist, album)."
)

collaborative_weight = st.sidebar.slider(
    "Collaborative Filtering Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Higher value means more recommendations based on what similar users like."
)

# Ensure weights sum to 1 (optional, but good practice for clarity)
if content_weight + collaborative_weight != 1.0:
    st.sidebar.warning(f"Weights sum to {content_weight + collaborative_weight:.1f}. Consider adjusting to sum to 1.0.")


popularity_penalty_weight = st.sidebar.slider(
    "Popularity Penalty Weight",
    min_value=0.0,
    max_value=0.5,
    value=0.1,
    step=0.05,
    help="Higher value penalizes popular songs more, promoting diversity."
)

if st.sidebar.button("Get Recommendations"):
    if not user_id_input:
        st.error("Please enter a User ID.")
    else:
        with st.spinner("Generating recommendations..."):
            try:
                recommendations = hybrid_recommendation(
                    user_id_input,
                    merged_df_with_features, # Use the merged_df with combined_features
                    lastfm_df,
                    user_similarity_matrix,
                    user_item_matrix,
                    content_similarity_matrix,
                    track_popularity,
                    content_weight=content_weight,
                    collaborative_weight=collaborative_weight,
                    popularity_penalty_weight=popularity_penalty_weight,
                    top_n=num_recommendations
                )

                if not recommendations.empty:
                    st.session_state.recommendations = recommendations.reset_index(drop=True)
                    st.session_state.current_song_index = 0
                    st.success(f"Generated {len(recommendations)} recommendations for '{user_id_input}'.")
                    # Rerun to update the player section
                    st.rerun()
                else:
                    st.info(f"No recommendations found for user '{user_id_input}'. Try a different user or adjust parameters.")
            except Exception as e:
                st.error(f"An error occurred during recommendation generation: {e}")
                st.warning("Please ensure the user ID exists in the dataset or check the data processing steps.")

st.markdown("---")

# --- Interactive Music Player Section ---
st.header("🎵 Recommended Music Player")

if not st.session_state.recommendations.empty:
    current_index = st.session_state.current_song_index
    total_songs = len(st.session_state.recommendations)

    if total_songs > 0:
        current_song = st.session_state.recommendations.iloc[current_index]

        # Generate a random color for the background
        # You can make this more sophisticated, e.g., based on song features
        random_color = f"#{random.randint(0, 0xFFFFFF):06x}"

        st.markdown(f"""
        <div style="
            background-color: {random_color};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        ">
            <h3>{current_song['Track']}</h3>
            <p><strong>Artist:</strong> {current_song['Artist']}</p>
            <p>Score: {current_song['final_score']:.4f}</p>
            <p>({current_index + 1} / {total_songs})</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("⏪ Previous Song", use_container_width=True):
                if st.session_state.current_song_index > 0:
                    st.session_state.current_song_index -= 1
                    st.rerun()
                else:
                    st.warning("You are at the first song.")

        with col3:
            if st.button("⏩ Next Song", use_container_width=True):
                if st.session_state.current_song_index < total_songs - 1:
                    st.session_state.current_song_index += 1
                    st.rerun()
                else:
                    st.warning("You are at the last song.")
        with col2:
            st.empty() # Placeholder for centering or future controls

    else:
        st.info("No songs to display. Generate recommendations first!")
else:
    st.info("Generate recommendations using the sidebar to start the music player.")


st.markdown("---")
st.markdown("### How it Works:")
st.markdown("""
This system uses a **Hybrid** approach:
- **Content-Based Filtering:** Recommends songs similar to what the user has liked, based on `Artist`, `Track`, `Album`, and `Genre`.
- **Collaborative Filtering:** Recommends songs based on what similar users like.
- **Popularity Penalty:** Reduces the score of very popular songs to encourage discovery of less common tracks.
You can adjust the weights to see how different components influence the recommendations!
""")

st.markdown("---")
st.markdown("### Data Preview (First 5 rows of Last.fm sample data)")
st.dataframe(lastfm_df.head())

st.markdown("### User-Item Matrix Preview (First 5x5 rows)")
st.dataframe(user_item_matrix.head())