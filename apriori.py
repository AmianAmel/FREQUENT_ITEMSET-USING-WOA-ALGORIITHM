from docx import Document
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import numpy as np

def load_transactions_from_docx(file_path):   
    transactions = []
    doc = Document(file_path)
    for para in doc.paragraphs:
        # Split the paragraph by spaces (as items are space-separated)
        items = para.text.split()
        # Clean up whitespace and ensure non-empty items
        items = [item.strip() for item in items if item.strip()]
        if items:
            transactions.append(items)
    return transactions

def create_one_hot_encoded_df(transactions):
    
    # Get unique items
    unique_items = sorted(list(set(item for sublist in transactions for item in sublist)))
    
    # Create DataFrame with boolean values
    encoded_data = []
    for transaction in transactions:
        row = {item: (item in transaction) for item in unique_items}
        encoded_data.append(row)
    
    return pd.DataFrame(encoded_data, dtype=bool)

def analyze_patterns(transactions, min_support=0.01, min_confidence=0.5, max_len=3):
    try:
        # Convert transactions to one-hot encoded DataFrame
        print("Converting transactions to DataFrame...")
        df = create_one_hot_encoded_df(transactions)
        
        # Find frequent itemsets with lower min_support and max_len limit
        print(f"Finding frequent itemsets (min_support={min_support}, max_len={max_len})...")
        frequent_itemsets = apriori(df,min_support=min_support,use_colnames=True,max_len=max_len)
        
        print("Generating association rules...")
        # Generate association rules with the frequent itemsets
        rules = association_rules(frequent_itemsets, 
                                metric="confidence",
                                min_threshold=min_confidence,
                                support_only=False)  # Include all metrics
        
        return frequent_itemsets, rules
    
    except MemoryError:
        print("Memory error occurred. Try increasing min_support or decreasing max_len.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
   
    file_path = 'chess(3).docx'
    
    # Load transactions
    transactions = load_transactions_from_docx(file_path)
    print(f"Loaded transactions: {len(transactions)}")
    
    # Analyze patterns with more conservative parameters
    frequent_itemsets, rules = analyze_patterns(
        transactions,
        min_support=0.05,  # 5% minimum support
        min_confidence=0.5,  # 50% minimum confidence
        max_len=3  # Maximum of 3 items in a set
    )
    
    # Display results
    if frequent_itemsets is not None:
        print("\nNumber of Frequent Itemsets found:", len(frequent_itemsets))
        print("\nFirst few Frequent Itemsets:")
        print(frequent_itemsets.head())
        
        if rules is not None and not rules.empty:
            print("\nNumber of Rules found:", len(rules))
            print("\nFirst few Association Rules:")
            print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
        else:
            print("\nNo rules were generated. Try adjusting the confidence threshold.")
    else:
        print("\nNo frequent itemsets were found. Try adjusting the support threshold.")
