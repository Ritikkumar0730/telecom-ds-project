from telco.rag import LSARetriever, Retriever, evaluate_retrieval, load_chunks, load_eval_questions


def test_chunks_have_doc_and_section_metadata():
    chunks = load_chunks()
    assert len(chunks) >= 10
    assert all(c.doc_id and c.section and c.text for c in chunks)
    assert {"prepaid_plans", "fraud_and_security"} <= {c.doc_id for c in chunks}


def test_tfidf_retrieval_recall_on_eval_set():
    result = evaluate_retrieval(Retriever(load_chunks()), load_eval_questions(), k=3)
    assert result["recall_at_3"] >= 0.9, result["misses"]


def test_lsa_retrieval_is_a_working_alternative():
    result = evaluate_retrieval(LSARetriever(load_chunks()), load_eval_questions(), k=3)
    assert result["recall_at_3"] >= 0.9, result["misses"]


def test_unrelated_query_returns_no_confident_match():
    retriever = Retriever(load_chunks())
    assert retriever.search("quantum chromodynamics lattice gauge", k=3) == []
