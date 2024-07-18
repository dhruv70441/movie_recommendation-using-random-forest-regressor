import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import tkinter as tk
from tkinter import messagebox

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv")

# Preprocess dataset
df_features = df[['Movie_Genre', 'Movie_Keywords', 'Movie_Tagline', 'Movie_Cast', 'Movie_Director']].fillna('')
X = df_features['Movie_Genre'] + ' ' + df_features['Movie_Keywords'] + ' ' + df_features['Movie_Tagline'] + ' ' + df_features['Movie_Cast'] + ' ' + df_features['Movie_Director']

# Feature extraction
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(X)

# Calculate similarity score
Similarity_Score = cosine_similarity(x)

# Function to get movie recommendations
def get_recommendations(movie_name):
    All_Movie_Title_List = df['Movie_Title'].tolist()
    Movie_Recommendation = difflib.get_close_matches(movie_name, All_Movie_Title_List)
    if not Movie_Recommendation:
        return []

    Close_Match = Movie_Recommendation[0]
    Index_of_Close_Match_Movie = df[df.Movie_Title == Close_Match]['Movie_ID'].values[0]
    Recommendation_Score = list(enumerate(Similarity_Score[Index_of_Close_Match_Movie]))
    Sorted_Similar_Movie = sorted(Recommendation_Score, key=lambda x: x[1], reverse=True)
    
    recommended_movies = []
    for i, movie in enumerate(Sorted_Similar_Movie):
        if i == 10:
            break
        index = movie[0]
        title_from_index = df[df.index == index]['Movie_Title'].values[0]
        recommended_movies.append(title_from_index)
    
    return recommended_movies

# GUI Application
class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation System")
        self.root.geometry("600x400")
        
        self.label = tk.Label(root, text="Enter your favourite movie name:", font=('Arial', 14))
        self.label.pack(pady=20)
        
        self.entry = tk.Entry(root, width=50, font=('Arial', 14))
        self.entry.pack(pady=10)
        
        self.button = tk.Button(root, text="Get Recommendations", command=self.recommend_movies, font=('Arial', 14), bg='blue', fg='white')
        self.button.pack(pady=20)
        
        self.result_label = tk.Label(root, text="Top 10 Movies Suggested for You:", font=('Arial', 14))
        self.result_label.pack(pady=20)
        
        self.result_box = tk.Listbox(root, width=80, height=10, font=('Arial', 12))
        self.result_box.pack(pady=10)
    
    def recommend_movies(self):
        self.result_box.delete(0, tk.END)
        movie_name = self.entry.get()
        recommendations = get_recommendations(movie_name)
        if not recommendations:
            messagebox.showerror("Error", "Movie not found!")
        else:
            for i, movie in enumerate(recommendations, 1):
                self.result_box.insert(tk.END, f"{i}. {movie}")

# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()
