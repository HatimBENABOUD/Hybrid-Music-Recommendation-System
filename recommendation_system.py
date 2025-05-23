import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Data Loading (Using your provided sample CSVs) ---
def load_sample_data():
    """
    Loads data from provided CSV files and augments Last.fm data for better
    demonstration of collaborative filtering if the sample is too small.
    Includes robust column cleaning and error checking.
    """
    try:
        # Load Last.fm sample data
        lastfm_df = pd.read_csv('last.fm_data_sample.csv')
        # Clean column names: strip whitespace and remove internal spaces
        lastfm_df.columns = lastfm_df.columns.str.strip()
        lastfm_df.columns = lastfm_df.columns.str.replace(' ', '') # Remove spaces within column names if any

        # Drop 'Unnamed: 0' if it exists, as it's typically an artifact of saving DataFrames with index
        if 'Unnamed: 0' in lastfm_df.columns:
            lastfm_df = lastfm_df.drop(columns=['Unnamed: 0'])

        # Load tracks sample data for genre information
        tracks_df = pd.read_csv('tracks_sample.csv')
        # Clean column names for tracks_df as well
        tracks_df.columns = tracks_df.columns.str.strip()
        tracks_df.columns = tracks_df.columns.str.replace(' ', '') # Remove spaces within column names if any
        if 'Unnamed: 0' in tracks_df.columns: # Check for unnamed column in tracks_df too
            tracks_df = tracks_df.drop(columns=['Unnamed: 0'])

        # Rename columns in tracks_df to match expected names for merging and feature creation
        # Based on error message, 'name' should be 'Track' and 'artists' should be 'Artist'
        tracks_df = tracks_df.rename(columns={'name': 'Track', 'artists': 'Artist'})

        # --- Data Augmentation for Collaborative Filtering Demonstration ---
        # If the sample has very few unique users, add some dummy users
        unique_users_in_sample = lastfm_df['Username'].nunique() if 'Username' in lastfm_df.columns else 0
        if unique_users_in_sample < 5: # Arbitrary threshold for adding dummy users
            print(f"Warning: Only {unique_users_in_sample} unique users in last.fm_data_sample.csv. Augmenting with dummy users.")
            dummy_data_lastfm = {
                'Username': ['User_A', 'User_A', 'User_A', 'User_B', 'User_B', 'User_C', 'User_C', 'User_D', 'User_D', 'User_E', 'User_E',
                             'User_A', 'User_B', 'User_C', 'User_D', 'User_E', 'User_F', 'User_F', 'User_G', 'User_G'],
                'Artist': ['Artist X', 'Artist Y', 'Artist Z', 'Artist X', 'Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist E', 'Artist F', 'Artist G',
                           'Artist I', 'Artist J', 'Artist K', 'Artist L', 'Artist M', 'Artist N', 'Artist O', 'Artist P', 'Artist Q'],
                'Track': ['Song 1', 'Song 2', 'Song 3', 'Song 1', 'Song 4', 'Song 5', 'Song 6', 'Song 7', 'Song 8', 'Song 9', 'Song 10',
                          'Song 12', 'Song 13', 'Song 14', 'Song 15', 'Song 16', 'Song 17', 'Song 18', 'Song 19', 'Song 20'],
                'Album': ['Album 1', 'Album 2', 'Album 3', 'Album 1', 'Album 4', 'Album 5', 'Album 6', 'Album 7', 'Album 8', 'Album 9', 'Album 10',
                          'Album 12', 'Album 13', 'Album 14', 'Album 15', 'Album 16', 'Album 17', 'Album 18', 'Album 19', 'Album 20'],
                'Date': ['01 Feb 2021', '01 Feb 2021', '01 Feb 2021', '02 Feb 2021', '02 Feb 2021', '03 Feb 2021', '03 Feb 2021', '04 Feb 2021', '04 Feb 2021', '05 Feb 2021', '05 Feb 2021',
                         '09 Feb 2021', '10 Feb 2021', '10 Feb 2021', '11 Feb 2021', '11 Feb 2021', '12 Feb 2021', '12 Feb 2021', '13 Feb 2021', '13 Feb 2021'],
                'Time': ['10:00', '10:05', '10:10', '11:00', '11:05', '12:00', '12:05', '13:00', '13:05', '14:00', '14:05',
                         '18:05', '19:00', '19:05', '20:00', '20:05', '21:00', '21:05', '22:00', '22:05']
            }
            lastfm_df_augmented = pd.concat([lastfm_df, pd.DataFrame(dummy_data_lastfm)], ignore_index=True)
            lastfm_df = lastfm_df_augmented

        # Ensure required columns exist after potential renaming
        required_lastfm_cols = ['Track', 'Artist', 'Username']
        for col in required_lastfm_cols:
            if col not in lastfm_df.columns:
                raise KeyError(f"Required column '{col}' not found in lastfm_df. Available columns: {lastfm_df.columns.tolist()}")

        # Only check for 'Track' and 'Artist' in tracks_df as 'Genre' is not in your sample
        required_tracks_cols = ['Track', 'Artist']
        for col in required_tracks_cols:
            if col not in tracks_df.columns:
                raise KeyError(f"Required column '{col}' not found in tracks_df after renaming. Available columns: {tracks_df.columns.tolist()}")

        # Merge to get a comprehensive track list.
        # Drop duplicates from lastfm_df before merging to ensure unique track-artist pairs.
        merged_df = pd.merge(lastfm_df.drop_duplicates(subset=['Track', 'Artist']), tracks_df, on=['Track', 'Artist'], how='left')

        # Add 'Genre' column to merged_df if it doesn't exist (as it's not in tracks_sample.csv)
        # Fill with 'Unknown' to allow content feature creation to proceed
        if 'Genre' not in merged_df.columns:
            merged_df['Genre'] = 'Unknown'

        return lastfm_df, merged_df

    except FileNotFoundError as e:
        print(f"Error: One or more sample data files not found. Please ensure 'last.fm_data_sample.csv' and 'tracks_sample.csv' are in the same directory.")
        print(f"Details: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except KeyError as e: # Catch KeyError specifically for column issues
        print(f"Error: Missing expected column: {e}. Please check your CSV files for correct column names (e.g., 'Track', 'Artist', 'Username', 'Genre').")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame(), pd.DataFrame()


# --- Preprocessing and Feature Engineering ---

def create_content_features(df):
    """
    Creates a combined feature string for content-based filtering,
    including Artist, Track, Album, and Genre.
    """
    # Fill NaN values to avoid errors during string concatenation
    df['Artist'] = df['Artist'].fillna('')
    df['Track'] = df['Track'].fillna('')
    df['Album'] = df['Album'].fillna('')
    df['Genre'] = df['Genre'].fillna('') # Assuming Genre is available in merged_df

    df['combined_features'] = df['Artist'] + ' ' + df['Track'] + ' ' + df['Album'] + ' ' + df['Genre']
    return df

def calculate_content_similarity(merged_df):
    """
    Calculates content-based similarity matrix using TF-IDF and Cosine Similarity.
    """
    # Ensure combined_features are created
    merged_df_with_features = create_content_features(merged_df.copy())

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(merged_df_with_features['combined_features'])

    # Calculate cosine similarity
    content_similarity_matrix = cosine_similarity(tfidf_matrix)
    return content_similarity_matrix, merged_df_with_features

def create_user_item_matrix(lastfm_df):
    """
    Creates a user-item interaction matrix (play counts).
    """
    # Use 'size' to count interactions (plays)
    user_item_matrix = lastfm_df.pivot_table(index='Username', columns='Track', aggfunc='size', fill_value=0)
    return user_item_matrix

def calculate_collaborative_similarity(user_item_matrix):
    """
    Calculates user-user collaborative similarity matrix.
    """
    # Calculate cosine similarity between users
    similarity_matrix = pd.DataFrame(cosine_similarity(user_item_matrix),
                                     index=user_item_matrix.index,
                                     columns=user_item_matrix.index)
    return similarity_matrix

def calculate_track_popularity(lastfm_df):
    """
    Calculates popularity of each track based on total play counts.
    """
    track_popularity = lastfm_df['Track'].value_counts().reset_index()
    track_popularity.columns = ['Track', 'PlayCount']
    # Normalize popularity to a 0-1 scale for easier weighting
    max_plays = track_popularity['PlayCount'].max()
    track_popularity['NormalizedPopularity'] = track_popularity['PlayCount'] / max_plays
    return track_popularity.set_index('Track')['NormalizedPopularity']

# --- Hybrid Recommendation Logic ---

def hybrid_recommendation(user_id, merged_df, lastfm_df, user_similarity_matrix,
                          user_item_matrix, content_similarity_matrix, track_popularity,
                          content_weight=0.5, collaborative_weight=0.5, popularity_penalty_weight=0.2, top_n=10):
    """
    Generates hybrid recommendations for a given user, incorporating popularity bias mitigation.

    Args:
        user_id (str): The ID of the user for whom to generate recommendations.
        merged_df (pd.DataFrame): DataFrame containing unique tracks and their metadata.
        lastfm_df (pd.DataFrame): DataFrame of user listening history.
        user_similarity_matrix (pd.DataFrame): User-user similarity matrix.
        user_item_matrix (pd.DataFrame): User-item interaction matrix.
        content_similarity_matrix (np.array): Content-based similarity matrix.
        track_popularity (pd.Series): Series of normalized track popularity scores.
        content_weight (float): Weight for content-based scores (0 to 1).
        collaborative_weight (float): Weight for collaborative scores (0 to 1).
        popularity_penalty_weight (float): Weight to penalize popular items (0 to 1).
        top_n (int): Number of top recommendations to return.

    Returns:
        pd.DataFrame: DataFrame of top N hybrid recommendations with scores.
    """
    if user_id not in user_item_matrix.index:
        # Cold start for new user: default to popularity-based or content-based
        print(f"User '{user_id}' not found in historical data. Recommending popular tracks.")
        # For a real system, you'd use a more sophisticated cold-start strategy here
        # e.g., ask for initial preferences, or recommend top N most popular tracks
        # For now, we'll just return popular tracks as a fallback
        popular_tracks = track_popularity.sort_values(ascending=False).head(top_n).index.tolist()
        # Ensure 'Artist' column is present for consistency in return DataFrame
        artists_for_popular = [merged_df[merged_df['Track'] == t]['Artist'].iloc[0] if t in merged_df['Track'].values else 'Unknown' for t in popular_tracks]
        return pd.DataFrame({'Track': popular_tracks, 'Artist': artists_for_popular, 'final_score': [1.0] * len(popular_tracks),
                             'ContentScore': [0.0] * len(popular_tracks), 'CollaborativeScore': [0.0] * len(popular_tracks), 'Popularity': [1.0] * len(popular_tracks)})


    # 1. Collaborative Filtering Scores
    # Get similar users
    similar_users = user_similarity_matrix[user_id].sort_values(ascending=False).index.tolist()
    similar_users = [u for u in similar_users if u != user_id] # Exclude self

    collaborative_scores = pd.Series(0.0, index=merged_df['Track'].unique())

    # Get tracks already listened to by the current user
    user_listened_tracks = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist()

    for other_user in similar_users:
        if other_user in user_item_matrix.index:
            # Get tracks listened to by the similar user
            other_user_tracks = user_item_matrix.loc[other_user][user_item_matrix.loc[other_user] > 0].index.tolist()
            # Calculate similarity weight
            similarity_weight = user_similarity_matrix.loc[user_id, other_user]

            for track in other_user_tracks:
                if track not in user_listened_tracks: # Only recommend unlistened tracks
                    # Accumulate scores based on similar users' listening habits
                    # Using a simple sum of similarity weights for now
                    collaborative_scores[track] += similarity_weight

    # Normalize collaborative scores to a 0-1 range
    if collaborative_scores.max() > 0:
        collaborative_scores = collaborative_scores / collaborative_scores.max()
    else:
        collaborative_scores = pd.Series(0.0, index=merged_df['Track'].unique())


    # 2. Content-Based Filtering Scores
    # Get the tracks the user has listened to
    user_tracks_df = merged_df[merged_df['Track'].isin(user_listened_tracks)]

    content_scores = pd.Series(0.0, index=merged_df['Track'].unique())

    if not user_tracks_df.empty:
        # Get indices of user's listened tracks in the merged_df
        # Ensure these indices correspond to the content_similarity_matrix
        # This requires mapping track names to their original indices in merged_df_with_features
        track_to_idx = {track: idx for idx, track in enumerate(merged_df['Track'].unique())}
        user_track_indices = [track_to_idx[t] for t in user_listened_tracks if t in track_to_idx]

        if user_track_indices: # Ensure there are valid indices
            for i, track_name in enumerate(merged_df['Track'].unique()):
                if track_name not in user_listened_tracks:
                    if track_name in track_to_idx:
                        track_idx = track_to_idx[track_name]
                        # Average similarity of this track to all tracks the user has listened to
                        avg_similarity = content_similarity_matrix[track_idx, user_track_indices].mean()
                        content_scores[track_name] = avg_similarity
        else:
            print(f"Warning: No valid indices found for user's listened tracks for content-based filtering for user {user_id}.")
            content_scores = pd.Series(0.0, index=merged_df['Track'].unique())
    else:
        # If user has no listened tracks for content-based, default to 0
        content_scores = pd.Series(0.0, index=merged_df['Track'].unique())

    # Normalize content scores to a 0-1 range
    if content_scores.max() > 0:
        content_scores = content_scores / content_scores.max()
    else:
        content_scores = pd.Series(0.0, index=merged_df['Track'].unique())


    # 3. Combine Scores (Weighted Hybrid) and Apply Popularity Penalty
    # Ensure all unique tracks from merged_df are considered for final scores
    all_unique_tracks = merged_df['Track'].unique()
    hybrid_scores_data = []
    for track_name in all_unique_tracks:
        artist_name = merged_df[merged_df['Track'] == track_name]['Artist'].iloc[0] if track_name in merged_df['Track'].values else 'Unknown'
        hybrid_scores_data.append({
            'Track': track_name,
            'Artist': artist_name,
            'ContentScore': content_scores.get(track_name, 0.0), # Use .get to handle tracks not in content_scores
            'CollaborativeScore': collaborative_scores.get(track_name, 0.0) # Use .get to handle tracks not in collaborative_scores
        })
    hybrid_scores = pd.DataFrame(hybrid_scores_data)

    # Add popularity score (0-1, 1 being most popular)
    hybrid_scores['Popularity'] = hybrid_scores['Track'].map(track_popularity).fillna(0)

    # Calculate final score
    hybrid_scores['final_score'] = (hybrid_scores['ContentScore'] * content_weight +
                                    hybrid_scores['CollaborativeScore'] * collaborative_weight)

    # Apply popularity penalty: reduce score for popular items.
    # A higher penalty_weight means more reduction for popular items.
    # We subtract (Popularity * penalty_weight) to penalize.
    # Ensure penalty doesn't make score negative or too low if not desired.
    hybrid_scores['final_score'] = hybrid_scores['final_score'] - (hybrid_scores['Popularity'] * popularity_penalty_weight)
    hybrid_scores['final_score'] = hybrid_scores['final_score'].clip(lower=0) # Ensure scores are not negative

    # Filter out tracks the user has already listened to
    hybrid_scores = hybrid_scores[~hybrid_scores['Track'].isin(user_listened_tracks)]

    # Sort and return top N recommendations
    top_recommendations = hybrid_scores.sort_values(by='final_score', ascending=False).head(top_n)

    return top_recommendations[['Track', 'Artist', 'final_score', 'ContentScore', 'CollaborativeScore', 'Popularity']]

# --- Main execution for testing (optional, for direct script run) ---
if __name__ == "__main__":
    print("Loading data...")
    lastfm_df, merged_df = load_sample_data()

    if lastfm_df.empty or merged_df.empty:
        print("Data loading failed. Exiting.")
    else:
        print("Calculating content similarity...")
        content_similarity_matrix, merged_df_with_features = calculate_content_similarity(merged_df)

        print("Creating user-item matrix...")
        user_item_matrix = create_user_item_matrix(lastfm_df)

        print("Calculating collaborative similarity...")
        user_similarity_matrix = calculate_collaborative_similarity(user_item_matrix)

        print("Calculating track popularity...")
        track_popularity = calculate_track_popularity(lastfm_df)

        # Example user IDs from the augmented data
        example_user_ids = ['Babs_05', 'User_A', 'User_B', 'User_C']

        for user_id_to_recommend in example_user_ids:
            print(f"\nGenerating hybrid recommendations for user: {user_id_to_recommend}")

            top_recommendations = hybrid_recommendation(
                user_id_to_recommend,
                merged_df_with_features, # Use the merged_df with combined_features
                lastfm_df,
                user_similarity_matrix,
                user_item_matrix,
                content_similarity_matrix,
                track_popularity,
                content_weight=0.6,
                collaborative_weight=0.4,
                popularity_penalty_weight=0.1,
                top_n=10
            )
            print(f"\n🎧 Top Hybrid Recommendations for {user_id_to_recommend}:")
            print(top_recommendations)

        user_id_new = 'NewUser123'
        print(f"\nGenerating hybrid recommendations for new user (cold start): {user_id_new}")
        top_recommendations_new = hybrid_recommendation(
            user_id_new,
            merged_df_with_features,
            lastfm_df,
            user_similarity_matrix,
            user_item_matrix,
            content_similarity_matrix,
            track_popularity,
            content_weight=0.6,
            collaborative_weight=0.4,
            popularity_penalty_weight=0.1,
            top_n=5
        )
        print("\n🎧 Top Hybrid Recommendations for New User:")
        print(top_recommendations_new)