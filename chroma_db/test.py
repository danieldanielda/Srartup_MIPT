# pip install chromadb

import chromadb
from chromadb.config import Settings

# Connect with no authentication
# client = chromadb.HttpClient(host='localhost', port=8000)

# Connect with role-based authentication
'''chroma_client = chromadb.HttpClient(host='localhost', port=8000,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_server_authn_provider="chromadb.auth.simple_rbac_authz.SimpleRBACAuthorizationProvider",
        chroma_client_auth_credentials="test-token-readonly"
    )
)'''

# Connect with token authentication
client = chromadb.HttpClient(host='localhost', port=8000,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials="test-token"
    )
)

print("Heartbeat:", client.heartbeat())

# test get or create collection
collection = client.get_or_create_collection("test_collection")

collection.add(
    embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    documents=["doc1", "doc2"],  # text                  
    ids=["id1", "id2"]                            
)
results = collection.query(
    query_embeddings=[[1.1, 2.1, 3.1]],  # запрос пользователя
    n_results=10                        # Get 2 closest
)

# get early created collection for test
#my_collection = client.get_or_create_collection("test_collection")

print("Search_result:")
print(results)
#print(my_collection)