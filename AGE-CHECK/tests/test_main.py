from datetime import date, timedelta

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["message"] == "Age Check API is running"


def test_ui_endpoint_returns_html() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Hey buddy" in response.text


def test_enter_endpoint_returns_html() -> None:
    response = client.get("/enter")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_check_age_success() -> None:
    response = client.post(
        "/age/check",
        json={"name": "Ravi", "date_of_birth": "2000-05-10"},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["name"] == "Ravi"
    assert isinstance(data["years"], int)
    assert isinstance(data["months"], int)
    assert isinstance(data["days"], int)
    assert isinstance(data["hours"], int)
    assert isinstance(data["minutes"], int)
    assert isinstance(data["seconds"], int)


def test_check_age_future_date() -> None:
    future_date = (date.today() + timedelta(days=1)).isoformat()
    response = client.post(
        "/age/check",
        json={"name": "Ravi", "date_of_birth": future_date},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "date_of_birth cannot be in the future"
