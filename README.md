# 🎶 Hybrid Music Recommendation System

This project implements a hybrid music recommendation system using a combination of **Content-Based Filtering** and **Collaborative Filtering**, powered by Streamlit for an interactive user interface. It also includes a mechanism to mitigate popularity bias, encouraging the discovery of less common tracks.

## 🌟 Features

* **Hybrid Recommendation Engine:** Combines two powerful recommendation techniques:
    * **Content-Based Filtering:** Recommends songs similar to what the user has liked, based on track metadata (Artist, Track, Album, Genre).
    * **Collaborative Filtering:** Recommends songs based on the listening habits of similar users.
* **Popularity Bias Mitigation:** Allows applying a penalty to highly popular tracks to promote diverse recommendations.
* **Interactive Streamlit UI:** A user-friendly web interface for:
    * Selecting or entering a user ID to get recommendations.
    * Adjusting the weights for content-based and collaborative filtering.
    * Setting the popularity penalty strength.
    * Controlling the number of recommendations.
* **"Music Player" Interface:** An interactive section to navigate through the recommended songs one by one, featuring a dynamic color display for each track (in lieu of album art).
* **Cold Start Handling:** Provides a fallback mechanism for new users (those not in the historical data), typically recommending popular tracks.

## ⚙️ How It Works

The system operates in several key steps:

1.  **Data Loading & Preprocessing:**
    * Loads listening history data (`last.fm_data_sample.csv`) and track metadata (`tracks_sample.csv`).
    * Handles potential column name discrepancies (e.g., renames 'name' to 'Track' and 'artists' to 'Artist' in `tracks_sample.csv`).
    * Augments sample data with dummy users if the original sample is too small for effective collaborative filtering.
    * Ensures consistent column structures, adding a 'Genre' column filled with 'Unknown' if not present in `tracks_sample.csv`.
2.  **Content-Based Features:**
    * Creates a combined text feature (Artist + Track + Album + Genre) for each song.
    * Uses `TF-IDF` (Term Frequency-Inverse Document Frequency) to vectorize these features.
    * Calculates **Cosine Similarity** between songs to determine content-based likeness.
3.  **Collaborative Filtering Features:**
    * Constructs a **User-Item Matrix** from the listening history, where values represent play counts.
    * Calculates **Cosine Similarity** between users to find users with similar listening patterns.
4.  **Track Popularity:**
    * Computes the popularity of each track based on its total play counts in the `lastfm_df`.
5.  **Hybrid Recommendation Logic:**
    * For a given user, it generates initial scores from both content-based and collaborative filtering components.
    * These scores are then combined using adjustable weights specified by the user in the Streamlit sidebar.
    * A **popularity penalty** is applied, which reduces the final score of more popular tracks based on the user-defined penalty weight.
    * Filters out tracks the user has already listened to.
    * Returns the top N unlistened recommendations based on the final hybrid score.

## 🚀 Installation & Setup

To run this project locally, follow these steps:

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <your_repository_url> # Replace with your actual repo URL if applicable
    cd <project_directory>
    ```
    (If not using Git, ensure you have `app.py`, `recommendation_system.py`, `last.fm_data_sample.csv`, and `tracks_sample.csv` in the same folder.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install the required libraries:**
    ```bash
    pip install streamlit pandas scikit-learn numpy
    ```

4.  **Ensure Sample Data is Present:**
    Make sure the files `last.fm_data_sample.csv` and `tracks_sample.csv` are in the same directory as your Python scripts.

## 🏃 Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Access the App:**
    Your web browser will automatically open the Streamlit application, usually at `http://localhost:8501`.

3.  **Interact with the Sidebar:**
    * **Select or Enter User ID:** Choose an existing user from the sample data or type in a new ID to test the cold start functionality.
    * **Number of Recommendations:** Adjust how many songs you want to be recommended.
    * **Recommendation Weights:** Use the sliders to control the influence of Content-Based vs. Collaborative Filtering.
    * **Popularity Penalty Weight:** Increase this value to get more diverse, less popular recommendations.
    * Click the **"Get Recommendations"** button.

4.  **Use the Music Player:**
    * Once recommendations are generated, an interactive "Recommended Music Player" section will appear.
    * Click **"⏪ Previous Song"** and **"⏩ Next Song"** to navigate through the recommendations.
    * Each song will be displayed with its title, artist, score, and a unique colored background.

## 📂 Project Structure

```
.
├── app.py                      # Main Streamlit application file
├── recommendation_system.py    # Core logic for data loading, preprocessing, and recommendation algorithms
├── last.fm_data_sample.csv     # Sample user listening history data
└── tracks_sample.csv           # Sample track metadata (including 'name' and 'artists' columns)
```

## 📝 Notes on Sample Data

* `last.fm_data_sample.csv` is expected to have columns like `Username`, `Artist`, `Track`, `Album`.
* `tracks_sample.csv` is expected to have columns like `name` (for `Track`) and `artists` (for `Artist`), and potentially other metadata. The current implementation handles the absence of a 'Genre' column by filling it with 'Unknown' for content-based features.
* The system includes augmentation logic to add dummy users if the `last.fm_data_sample.csv` contains too few unique users, which is helpful for demonstrating collaborative filtering with small datasets.
