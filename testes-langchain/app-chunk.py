#https://pub.towardsai.net/rag-in-production-chunking-decisions-96a214dbbdc6

import fitz  # PyMuPDF

# Open the PDF file
pdf_path = '/Users/tmedeirosb/Desktop/DEV/LANGCHAIN/ex-langchain/testes-langchain/data/OrganizacaoDidatica_2012_versaoFINAL_20mai2012 (4).pdf'
doc = fitz.open(pdf_path)

# Extract text from the first page
text = ""
for page in doc:
    text += page.get_text()

# Close the document
doc.close()

# Display the beginning of the extracted text
#print(text)  # Show a portion to verify extraction

def naive_chunking(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


chunks = naive_chunking(text, chunk_size=200)

for i, chunk in enumerate(chunks[:3]):  # Display the first 3 chunks
    print(f"Chunk {i+1}:\n{'-'*10}\n{chunk}\n")
    print(f"{'-'*100}\n{'-'*100}\n") 


import nltk # Using NLTK 
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def sentence_chunking(text):
    sentences = sent_tokenize(text)
    return sentences


chunks = sentence_chunking(text)

for i, chunk in enumerate(chunks[:3]):  # Display the first 3 chunks
    print(f"Chunk {i+1}:\n{'-'*10}\n{chunk}\n")
    print(f"{'-'*100}\n{'-'*100}\n")

