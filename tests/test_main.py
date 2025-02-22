import pytest
from fastapi.testclient import TestClient

# FIX THIS IMPORT
from app.app import app 

client = TestClient(app)

def test_add_doc():
    response = client.post("/add-doc/?new_doc=Deep learning is a powerful AI technique.")
    assert response.status_code == 200
    assert "Corpus added" in response.json()


def test_get_query():
    response = client.get("/query/")
    assert response.status_code == 200
    assert "bm25 vectors" in response.json()


def test_reset_corpus():
    response = client.post("/reset-corpus/?delete_all=Y")
    assert response.status_code == 200
    assert response.json() == {"message": "Corpus reset"}
    

def test_find_similar():
    client.post("/add-doc/", params={"new_doc": "Artificial Intelligence is transforming industries."})

  
    response = client.post("/find-similar/", params={"query": "Artificial Intelligence"})
    assert response.status_code == 200
    assert "most similar result" in response.json()
