import numpy as np
from docx import Document
import pandas as pd

def initialize_population(nsols, num_items):
    return np.random.rand(nsols, num_items)

def measure_states(quantum_states):
    return (quantum_states > 0.5).astype(int)

def fitness(solution, binary_matrix, minsup):
    selected_items = np.where(solution == 1)[0]
    if len(selected_items) == 0:
        return 0
    support_count = np.sum(np.all(binary_matrix[:, selected_items] == 1, axis=1))
    print(f"Selected Items: {selected_items}, Support Count: {support_count}")
    return support_count if support_count >= minsup else 0

def rank_solutions(solutions, binary_matrix, minsup):
    fitness_values = [fitness(sol, binary_matrix, minsup) for sol in solutions]
    ranked_solutions = sorted(zip(fitness_values, solutions), key=lambda x: x[0], reverse=True)
    return ranked_solutions

def update_solution(sol, best_sol, b, a, num_items):
    r = np.random.rand()
    A = 2 * a * r - a
    C = 2 * r
    if np.random.rand() < 0.5:
        D = np.abs(C * best_sol - sol)
        new_sol = best_sol - A * D
    else:
        L = np.random.uniform(-1, 1)
        distance_to_best = np.abs(best_sol - sol)
        new_sol = distance_to_best * np.exp(b * L) * np.cos(2 * np.pi * L) + best_sol
    return np.clip(new_sol, 0, 1)

def quantum_whale_optimization(binary_matrix, items, minsup, nsols=10, b=1, a=2, a_step=0.01, generations=100):
    num_items = len(items)
    quantum_states = initialize_population(nsols, num_items)
    for generation in range(generations):
        solutions = measure_states(quantum_states)
        ranked_solutions = rank_solutions(solutions, binary_matrix, minsup)
        best_solution = ranked_solutions[0][1]
        for i in range(nsols):
            quantum_states[i] = update_solution(solutions[i], best_solution, b, a, num_items)
        a = max(a - a_step, 0)
    best_fitness, best_solution = rank_solutions(measure_states(quantum_states), binary_matrix, minsup)[0]
    best_itemset = [items[i] for i in range(num_items) if best_solution[i] == 1]
    return best_itemset, best_fitness

def load_and_preprocess_docx(file_path):
    """
    Read data from a DOCX file and convert it to a binary matrix.
    Assumes each row in the document is a space-separated sequence of 1s and 0s.
    """
    doc = Document(file_path)
    rows = []
    for paragraph in doc.paragraphs:
        # Skip empty paragraphs
        if paragraph.text.strip():
            # Convert space-separated string of 1s and 0s to list of integers
            row = [int(bit) for bit in paragraph.text.strip().split()]
            rows.append(row)
    
    # Convert to numpy array
    binary_matrix = np.array(rows)
    
    # Generate column names (items)
    items = [f"Item_{i}" for i in range(binary_matrix.shape[1])]
    
    return binary_matrix, items

def main():
    file_path = "chess.docx"  # Your DOCX file path
    minsup = 3
    
    try:
        binary_matrix, items = load_and_preprocess_docx(file_path)
        best_itemset, best_fitness = quantum_whale_optimization(binary_matrix, items, minsup)
        print("Best Itemset:", best_itemset)
        print("Support Count:", best_fitness)
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()