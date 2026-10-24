variable "project_id" {
  description = "ID do projeto GCP"
  type        = string
}

variable "region" {
  description = "Região do Cloud Run. us-central1 = tarifa Tier 1 (mais barata); southamerica-east1 (São Paulo) é Tier 2."
  type        = string
  default     = "us-central1"
}

variable "billing_account" {
  description = "ID da conta de faturamento (XXXXXX-XXXXXX-XXXXXX), usado no alerta de orçamento"
  type        = string
}

variable "budget_amount" {
  description = "Valor do orçamento mensal, na moeda da conta de faturamento (ex.: BRL)"
  type        = number
  default     = 10
}

variable "alert_email" {
  description = "E-mail que recebe os alertas de orçamento e latência (opcional)"
  type        = string
  default     = ""
}

variable "image_tag" {
  description = "Tag da imagem da API no Artifact Registry"
  type        = string
  default     = "latest"
}

variable "min_instances" {
  description = "Mínimo de instâncias. 0 = escala a zero sem tráfego (custo zero ocioso, com cold start)"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Máximo de instâncias: teto de custo do autoscaling"
  type        = number
  default     = 5
}

variable "concurrency" {
  description = "Requisições simultâneas por instância antes de o Cloud Run criar outra"
  type        = number
  default     = 40
}

variable "public" {
  description = "Permite acesso sem autenticação (demonstração). Em produção, use false + IAM/IAP."
  type        = bool
  default     = true
}
