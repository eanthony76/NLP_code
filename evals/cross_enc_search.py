from sentence_transformers import CrossEncoder
import faiss
import numpy as np
import time

'''Helper function which fetches article info for query-article match. Right now it is formatted to pull the body 
column from the data dataframe, but can be adjusted'''
def cross_fetch_article_info(dataframe_idx, data):
    info = data.iloc[dataframe_idx]
    meta_dict = dict()
#     meta_dict['Title'] = info['Title']
    meta_dict['Body'] = info['body']
    return info['body']
    
'''Helper function which encodes the query using the BERT model and then performs a search to try to match the query vector
to the top k most similar articles'''
def cross_search(query, model, data):
    #read index and set top K at 15 best retrievals
    index = faiss.read_index('body_paragraphs.index')
    top_k = 15
    t=time.time()
    #Use retrieval model to encode
    query_vector = model.encode([query])
    #search vectorstore for top results
    distances, top_k = index.search(query_vector, top_k)
    print('>>>> Time to return results: {}'.format(time.time()-t))
    top_k_ids = top_k.tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results = [cross_fetch_article_info(idx, data=data) for idx in top_k_ids]
    #define cross-encoder model
    cross_model = CrossEncoder('ms-marco-MiniLM-L-12-v2', num_labels=1)
    #Re-rank results using cross-encoder
    cross_inp = [[query, cross_fetch_article_info(idx, data=data)] for idx in top_k_ids]
    cross_scores = cross_model.predict(cross_inp)
    #sort results
    results_list = []
    for idx in range(len(cross_scores)):
        results_list.append([top_k_ids[idx],cross_scores[idx]])
    ranked_results_idx = sorted(results_list, key=lambda x: x[1], reverse = True)
    print(ranked_results_idx)
    results = []
    results.append([[idx[1],cross_fetch_article_info(idx,data=data)] for idx in ranked_results_idx[0:4]])
    return results