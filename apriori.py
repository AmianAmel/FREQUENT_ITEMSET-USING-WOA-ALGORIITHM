from docx import Document

def load_transactions_from_docx(file_path):
    transactions = []
    doc = Document(file_path)
    for para in doc.paragraphs:
        # Split the paragraph by spaces (as items are space-separated)
        items = para.text.split()  # Splitting by spaces
        items = [item.strip() for item in items if item.strip()]  # Clean up whitespace
        if items:
            transactions.append(items)
    return transactions
file_path = 'chess(3).docx'  # Update this to the actual file path
transactions = load_transactions_from_docx(file_path)
print(transactions)



