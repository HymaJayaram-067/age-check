# Age Check API - Local CI/CD Setup

This project is a FastAPI app with local DevOps flow:

- Jenkins pipeline
- Unit testing with pytest
- SonarQube code analysis
- Docker image build
- ArgoCD deploy to Minikube

## 1) Run app locally (without Kubernetes)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Landing: `http://127.0.0.1:8000/`
Enter Details: `http://127.0.0.1:8000/enter`
Result Page: `http://127.0.0.1:8000/result` (opens after Calculate)
API Docs: `http://127.0.0.1:8000/docs`
Health: `http://127.0.0.1:8000/health`

## 2) Run unit tests

```powershell
pytest --junitxml=pytest-report.xml
```

## 3) Build Docker image

```powershell
docker build -t age-check-api:latest .
```

## 4) Deploy to Minikube manually

```powershell
minikube start
minikube image build -t age-check-api:latest .
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/ollama-deployment.yaml
kubectl apply -f k8s/ollama-service.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Get service URL:

```powershell
minikube service age-check-api-service -n age-check --url
```

### AI Plan (LLM) note
This project uses **local free LLM** via **Ollama** (running inside Minikube). If the model is not ready yet, the first click may take a few minutes while Ollama downloads it.


## 5) ArgoCD setup

Install ArgoCD in your Minikube cluster, then apply application:

```powershell
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl apply -f argocd/application.yaml
```

Important: update `repoURL` in `argocd/application.yaml` to your Git repo URL.

## 6) Jenkins setup

Use Jenkins pipeline job pointing to this repository and `Jenkinsfile`.

Required tools on Jenkins node:

- Python 3.11+
- Docker
- Minikube
- kubectl
- sonar-scanner

Required Jenkins configuration:

- SonarQube server name should be: `sonarqube-server`
- Jenkins should have access to Docker and Minikube

## 7) What your team can check

- Jenkins build stages (test, sonar, docker, deploy)
- SonarQube quality results for the Python code
- ArgoCD application sync status
- API endpoint from Minikube service URL

Example API request:

```bash
curl -X POST "http://127.0.0.1:8000/age/check" -H "Content-Type: application/json" -d "{\"name\":\"Ravi\",\"date_of_birth\":\"2000-05-10\"}"
```
