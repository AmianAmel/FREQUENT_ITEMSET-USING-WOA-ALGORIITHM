import numpy as np
import random
from collections import defaultdict
from docx import Document


def read_transactions(file_path):
    """
    Reads transactions from a .docx file and returns a list of transactions.
    Each transaction is a set of integers.
    """
    doc = Document(file_path)
    transactions = []
    for para in doc.paragraphs:
        if para.text.strip():
            numbers = {int(num) for num in para.text.strip().split()}
            if numbers:
                transactions.append(numbers)
    return transactions


def find_patterns_woa(transactions, min_support, max_size, n_whales=20, max_iter=50):
    """
    Implements the Whale Optimization Algorithm (WOA) to find the best itemset in a transaction dataset.
    """
    transaction_sets = [set(t) for t in transactions]
    n_transactions = len(transaction_sets)

    # Count the frequency of individual items
    item_counts = defaultdict(int)
    for transaction in transaction_sets:
        for item in transaction:
            item_counts[item] += 1

    # Filter items by minimum support
    frequent_items = [
        [item] for item, count in item_counts.items()
        if count / n_transactions >= min_support
    ]

    # Check if there are any frequent items
    if not frequent_items:
        print("No frequent items found with the given min_support.")
        return None

    # Initialize whale population (positions) with consistent size
    whales = [random.sample([item[0] for item in frequent_items], min(max_size, len(frequent_items))) for _ in range(n_whales)]

    # Initialize best solution
    best_whale = None
    best_fitness = 0

    # WOA parameters
    a = 2  # linearly decreases from 2 to 0
    b = 1  # constant for spiral motion

    # Optimization loop
    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)  # Decrease a linearly from 2 to 0

        for i, whale in enumerate(whales):
            # Compute fitness (support × size)
            support = sum(1 for t in transaction_sets if set(whale).issubset(t)) / n_transactions
            fitness = len(whale) * support

            # Update the best solution
            if fitness > best_fitness:
                best_fitness = fitness
                best_whale = whale

            # Update whale positions
            r1 = random.random()  # Random value in [0, 1]
            r2 = random.random()  # Random value in [0, 1]
            A = 2 * a * r1 - a    # Calculate A
            C = 2 * r2            # Calculate C

            if abs(A) < 1:  # Exploitation phase
                # Encircle prey (best whale)
                D = [abs(C * best_whale[j % len(best_whale)] - whale[j % len(whale)]) for j in range(max_size)]
                whales[i] = [(best_whale[j % len(best_whale)] - A * D[j]) for j in range(max_size)]
            else:  # Exploration phase
                # Randomly select a whale (not necessarily the best)
                random_whale = random.choice(whales)
                D = [abs(C * random_whale[j % len(random_whale)] - whale[j % len(whale)]) for j in range(max_size)]
                whales[i] = [(random_whale[j % len(random_whale)] - A * D[j]) for j in range(max_size)]

            # Spiral update (simulate bubble-net feeding)
            p = random.random()
            if p < 0.5:
                l = random.uniform(-1, 1)  # Random value in [-1, 1]
                D_prime = [abs(best_whale[j % len(best_whale)] - whale[j % len(whale)]) for j in range(max_size)]
                whales[i] = [
                    int(D_prime[j] * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale[j % len(best_whale)])
                    for j in range(max_size)
                ]

    # Return the best solution found
    return (best_whale, best_fitness)


# Read transactions from your chess.docx file
transactions = read_transactions("chess(3).docx")

# Find the best itemset with its fitness using WOA
best_item_fitness = find_patterns_woa(transactions, min_support=0.2, max_size=5)

# Print the best itemset with its fitness
if best_item_fitness:
    print('Best itemset with its fitness score is:', best_item_fitness)
else:
    print("No valid itemsets found.")
