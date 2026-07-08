from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)

def test_terrain_endpoint():
    response = client.get("/api/terrain")
    assert response.status_code == 200
    assert "image/png" in response.json()["image"]

def test_websocket_simulation():
    with client.websocket_connect("/ws/simulate") as websocket:
        data = websocket.receive_json()
        assert "step" in data
        assert "sensors" in data
        assert len(data["sensors"]) == 50
