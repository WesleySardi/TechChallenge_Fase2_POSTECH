#!/usr/bin/env bash
# Deploy da API no Google Cloud Run (bônus de nuvem).
# Pré-requisitos: gcloud autenticado (gcloud auth login && gcloud auth application-default login),
# Docker rodando, Terraform >= 1.6 e infra/terraform/terraform.tfvars preenchido.
set -euo pipefail
cd "$(dirname "$0")/../infra/terraform"

PROJECT=$(sed -n 's/^project_id *= *"\(.*\)".*/\1/p' terraform.tfvars)
REGION=$(sed -n 's/^region *= *"\(.*\)".*/\1/p' terraform.tfvars)
REGION=${REGION:-us-central1}
TAG=${1:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)}

terraform init -input=false

echo ">> 1/3 Habilitando APIs e criando o Artifact Registry..."
terraform apply -input=false -auto-approve \
  -target=google_project_service.apis -target=google_artifact_registry_repository.api

echo ">> 2/3 Build e push da imagem ($TAG)..."
IMAGE="$REGION-docker.pkg.dev/$PROJECT/tech-challenge/diagnostico-api:$TAG"
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
docker build --platform linux/amd64 -t "$IMAGE" ../..
docker push "$IMAGE"

echo ">> 3/3 Cloud Run + orçamento + alertas..."
terraform apply -input=false -auto-approve -var "image_tag=$TAG"
echo ">> API: $(terraform output -raw api_url)/docs"
echo ">> Para remover tudo e zerar custos: cd infra/terraform && terraform destroy"
