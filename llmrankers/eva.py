import numpy as np

def dcg_at_k(relevance_scores, k):
    """
    Calculate DCG (Discounted Cumulative Gain) at rank k.
    """
    relevance_scores = np.array(relevance_scores)[:k]
    if relevance_scores.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance_scores.size + 2))
    return np.sum(relevance_scores / discounts)

def ndcg_at_k(predicted_ranking, true_relevance, k=10):
    """
    Calculate NDCG at rank k.
    Args:
        predicted_ranking (list): Ranked list of document IDs (or indices) based on your algorithm.
        true_relevance (dict): A dictionary mapping document IDs (or indices) to their relevance scores.
        k (int): Rank at which to calculate NDCG.
    Returns:
        float: NDCG@k value.
    """
    # Get the relevance scores for the predicted ranking
    relevance_scores = [true_relevance.get(doc_id, 0) for doc_id in predicted_ranking]

    # Calculate DCG for the predicted ranking
    dcg = dcg_at_k(relevance_scores, k)

    # Sort true_relevance by score in descending order to get the ideal ranking
    ideal_ranking = sorted(true_relevance.values(), reverse=True)
    
    # Calculate IDCG for the ideal ranking
    idcg = dcg_at_k(ideal_ranking, k)

    # Avoid division by zero
    return dcg / idcg if idcg > 0 else 0.0

# Example Setup
if __name__ == "__main__":
    # User input (for reference)
    user_input = "What is the capital of France?"

    # List of output texts with their algorithm's relevance ratings
    algorithm_outputs = [
        (1, "The Eiffel Tower is located in Paris.", 2),  # (ID, Text, Rating)
        (2, "Paris is the capital of France.", 3),
        (3, "France is a country in Europe.", 2),
        (4, "Madrid is the capital of Spain.", 0),
        (5, "Berlin is the capital of Germany.", 0),
    ]

    # Predicted ranking based on the algorithm (IDs in ranked order)
    predicted_ranking = [1, 2, 3, 4, 5]

    # Algorithm's ratings (used as relevance scores)
    algorithm_ratings = {doc_id: rating for doc_id, _, rating in algorithm_outputs}

    # Print the output for verification
    print("User Input:", user_input)
    print("\nAlgorithm Outputs and Ratings:")
    for doc_id, text, rating in algorithm_outputs:
        print(f"ID: {doc_id}, Text: \"{text}\", Rating: {rating}")

    # Calculate NDCG@10
    ndcg_score = ndcg_at_k(predicted_ranking, algorithm_ratings, k=10)
    print(f"\nNDCG@10: {ndcg_score:.4f}")
