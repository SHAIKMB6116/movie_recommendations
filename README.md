# movie_recommendations
recommendation of movies to the users based on their reviews of movies given by them.
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-movie ratings
data = {
    'User': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Movie': ['Avengers', 'Titanic', 'Avengers', 'Gravity', 'Titanic', 'Gravity'],
    'Rating': [5, 4, 5, 3, 4, 5]
}
df = pd.DataFrame(data)

# Create user-movie rating matrix
matrix = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)

# Compute similarity between users
similarity = cosine_similarity(matrix)

# Recommend movies for user 'A' based on similar users
def recommend(user_id, matrix, similarity):
    user_idx = list(matrix.index).index(user_id)
    sim_scores = similarity[user_idx]
    similar_users = [matrix.index[i] for i in range(len(sim_scores)) if sim_scores[i] > 0.5 and i != user_idx]

    # Movies watched by similar users but not by current user
    user_movies = set(matrix.loc[user_id][matrix.loc[user_id] > 0].index)
    recommendations = set()
    for u in similar_users:
        movies = set(matrix.loc[u][matrix.loc[u] > 0].index)
        recommendations.update(movies - user_movies)
    return list(recommendations)

print("Recommendations for user A:", recommend('A', matrix, similarity))
