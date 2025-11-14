"""
Recommender System Explainability.

Explains collaborative filtering, matrix factorization, and recommendations.

Author: XPLIA Team
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

class CollaborativeFilteringExplainer:
    """Explain collaborative filtering recommendations."""

    def __init__(self, user_item_matrix: np.ndarray):
        self.user_item_matrix = user_item_matrix

    def explain_recommendation(
        self,
        user_id: int,
        item_id: int,
        k_similar_users: int = 5
    ) -> Dict[str, Any]:
        """Explain why item recommended to user."""

        # Find similar users
        user_vector = self.user_item_matrix[user_id]
        similarities = []

        for uid in range(len(self.user_item_matrix)):
            if uid != user_id:
                other_vector = self.user_item_matrix[uid]
                sim = float(np.dot(user_vector, other_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(other_vector) + 1e-8))
                similarities.append((uid, sim))

        # Top-k similar users
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = similarities[:k_similar_users]

        # Check if similar users liked this item
        evidence = []
        for uid, sim in similar_users:
            rating = self.user_item_matrix[uid, item_id]
            if rating > 0:
                evidence.append({
                    'similar_user_id': uid,
                    'similarity': sim,
                    'rating': float(rating)
                })

        return {
            'user_id': user_id,
            'item_id': item_id,
            'similar_users_who_liked': evidence,
            'explanation': f"{len(evidence)} similar users rated this item highly"
        }

class MatrixFactorizationExplainer:
    """Explain matrix factorization recommendations."""

    def __init__(self, user_factors: np.ndarray, item_factors: np.ndarray):
        self.user_factors = user_factors
        self.item_factors = item_factors

    def explain_prediction(self, user_id: int, item_id: int) -> Dict[str, Any]:
        """Explain predicted rating using latent factors."""

        user_vec = self.user_factors[user_id]
        item_vec = self.item_factors[item_id]

        # Prediction = dot product
        prediction = float(np.dot(user_vec, item_vec))

        # Factor contributions
        factor_contributions = user_vec * item_vec

        # Top contributing factors
        top_factors = np.argsort(np.abs(factor_contributions))[-3:][::-1]

        return {
            'predicted_rating': prediction,
            'user_factors': user_vec.tolist(),
            'item_factors': item_vec.tolist(),
            'factor_contributions': factor_contributions.tolist(),
            'top_contributing_factors': top_factors.tolist()
        }

# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Recommender System Explainability - Example")
    print("=" * 80)

    # User-item matrix (users x items)
    n_users, n_items = 100, 50
    user_item_matrix = np.random.randint(0, 6, (n_users, n_items))

    print("\n1. COLLABORATIVE FILTERING EXPLANATION")
    print("-" * 80)

    cf_exp = CollaborativeFilteringExplainer(user_item_matrix)
    exp = cf_exp.explain_recommendation(user_id=5, item_id=10, k_similar_users=5)

    print(f"Why recommend item {exp['item_id']} to user {exp['user_id']}?")
    print(f"Explanation: {exp['explanation']}")
    print(f"\nEvidence from similar users:")
    for evidence in exp['similar_users_who_liked'][:3]:
        print(f"  User {evidence['similar_user_id']} (similarity: {evidence['similarity']:.3f}) rated {evidence['rating']:.1f}")

    print("\n2. MATRIX FACTORIZATION EXPLANATION")
    print("-" * 80)

    k = 20  # Latent factors
    user_factors = np.random.randn(n_users, k)
    item_factors = np.random.randn(n_items, k)

    mf_exp = MatrixFactorizationExplainer(user_factors, item_factors)
    mf_result = mf_exp.explain_prediction(user_id=5, item_id=10)

    print(f"Predicted rating: {mf_result['predicted_rating']:.2f}")
    print(f"Top contributing latent factors: {mf_result['top_contributing_factors']}")

    print("\n" + "=" * 80)
