#!/usr/bin/env bash
# Demonstração de escalabilidade automática (HPA) num Kubernetes local com kind.
# Pré-requisitos: Docker rodando, kind e kubectl instalados, `python main_ga.py` já executado.
#
#   bash scripts/k8s_demo.sh            # cria cluster, implanta, gera carga e mostra o HPA escalando
#   kind delete cluster --name tech-challenge   # limpa tudo
set -euo pipefail
cd "$(dirname "$0")/.."
CLUSTER=tech-challenge

if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --wait 120s
  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
  # kind usa certificados autoassinados no kubelet
  kubectl -n kube-system patch deployment metrics-server --type=json \
    -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
  kubectl -n kube-system rollout status deployment/metrics-server --timeout=180s
fi

echo ">> Build e carga da imagem no cluster..."
docker build -t diagnostico-api:latest .
kind load docker-image diagnostico-api:latest --name "$CLUSTER"

echo ">> Deploy (Deployment + Service + HPA)..."
kubectl apply -f deploy/k8s/
kubectl rollout restart deployment/diagnostico-api
kubectl rollout status deployment/diagnostico-api --timeout=180s

echo ">> Gerando carga DENTRO do cluster (passa pelo balanceamento do Service)..."
kubectl delete pod loadgen --ignore-not-found
kubectl run loadgen --image=diagnostico-api:latest --image-pull-policy=IfNotPresent --restart=Never -- \
  python scripts/load_test.py --url http://diagnostico-api:8000 --stages 4:30 32:240 4:30

echo ">> Acompanhe (Ctrl+C para sair): réplicas sobem com a CPU e descem depois do pico"
kubectl get hpa diagnostico-api --watch
