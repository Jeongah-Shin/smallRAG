def rag_pipeline(query, retriever, generator, top_k=2):

    # Step 1 : Retrieve documents
    query_vec = retriever.transform([query])
    consine_sim = np.dot(query_vec, doc_vectors.T).toarray()
    top_docs_indices = consine_sim[0].argsort()[-top_k:][::-1]
    top_docs = [documents[i] for i in top_docs_indices]

    # Step 2 : Generator Input
    combined_input = " ".join(top_docs) + " " + query
    input_ids = tokenizer(combined_input, return_tensors="tf").input_ids

    # Generator Response
    outputs = generator.generate(input_ids, max_length=50, num_beams=3, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

response = rag_pipeline("Tell me about Elmo", vectorizer, model)
print("RAG Responses: ", response)