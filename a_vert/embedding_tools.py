import numpy as np
import requests
import json

    

def embedding_call(text, tei_endpoint):
    """Calls the Text-Embedding-Inference endpoint and return the embeddings
    array
    """
    payload = {"inputs": text}
    headers = {"Content-Type": "application/json"}
    response = requests.post(tei_endpoint, data=json.dumps(payload), headers=headers)
    embedding = np.array(json.loads(response.text))
    return embedding


def get_embedding(text, tei_endpoint, max_batch_size=32):
    """Call the Text-Embedding-Inference endpoint handling the endpoint batch 
    size.
    """
    if isinstance(text, list):
        if len(text) > max_batch_size:
            partial = embedding_call(text[:max_batch_size], tei_endpoint)
            last = max_batch_size
            while last < len(text):
                if last+max_batch_size > len(text):
                    part = embedding_call(text[last:], tei_endpoint)
                else:
                    part = embedding_call(text[last:last+max_batch_size], tei_endpoint)
                last += max_batch_size
                partial = np.concatenate([partial, part], axis=0)
            return partial
                
        else:
            return embedding_call(text, tei_endpoint)
    else:
        return embedding_call(text, tei_endpoint)

    
