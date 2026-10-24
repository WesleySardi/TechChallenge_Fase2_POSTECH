# Infraestrutura da API de diagnóstico no Google Cloud:
#   Artifact Registry (imagem) + Cloud Run (API com autoscaling) + orçamento + alerta de latência.
# A LLM local (Ollama/GPU) NÃO é implantada na nuvem: GPU gerenciada é cara. Na nuvem a
# API serve predição + fatores SHAP (LLM_ENABLED=0); a explicação por LLM roda localmente
# ou pode apontar para uma API externa (LLM_PROVIDER=openai).

locals {
  services = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "billingbudgets.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
  ]
  image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.api.repository_id}/diagnostico-api:${var.image_tag}"
}

data "google_project" "this" {}

resource "google_project_service" "apis" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "api" {
  location      = var.region
  repository_id = "tech-challenge"
  format        = "DOCKER"
  description   = "Imagens da API de diagnóstico (Tech Challenge Fase 2)"

  # Mantém só as 3 imagens mais recentes (armazenamento acima de 0,5 GB é cobrado)
  cleanup_policies {
    id     = "manter-recentes"
    action = "KEEP"
    most_recent_versions {
      keep_count = 3
    }
  }
  cleanup_policies {
    id     = "apagar-antigas"
    action = "DELETE"
    condition {
      older_than = "604800s"
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_service_account" "api" {
  account_id   = "diagnostico-api"
  display_name = "Cloud Run - API de diagnostico"
}

resource "google_cloud_run_v2_service" "api" {
  name                = "diagnostico-api"
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    service_account                  = google_service_account.api.email
    max_instance_request_concurrency = var.concurrency
    timeout                          = "60s"

    # Escalabilidade automática: o Cloud Run cria/remove instâncias conforme a
    # concorrência de requisições, entre min e max.
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = local.image
      ports {
        container_port = 8080
      }
      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle          = true # CPU só é cobrada durante requisições
        startup_cpu_boost = true # acelera o cold start
      }
      env {
        name  = "LLM_ENABLED"
        value = "0"
      }
      env {
        name  = "LOG_FORMAT"
        value = "json" # Cloud Logging indexa os campos do JSON
      }
      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 12
      }
      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.public ? 1 : 0
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------- custos

resource "google_monitoring_notification_channel" "email" {
  count        = var.alert_email == "" ? 0 : 1
  display_name = "Alertas Tech Challenge"
  type         = "email"
  labels = {
    email_address = var.alert_email
  }
  depends_on = [google_project_service.apis]
}

resource "google_billing_budget" "budget" {
  billing_account = var.billing_account
  display_name    = "Tech Challenge Fase 2"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }

  amount {
    specified_amount {
      units = tostring(var.budget_amount)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  dynamic "all_updates_rule" {
    for_each = var.alert_email == "" ? [] : [1]
    content {
      monitoring_notification_channels = [google_monitoring_notification_channel.email[0].id]
      disable_default_iam_recipients   = false
    }
  }

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------- monitoramento

# Alerta: latência p95 acima de 1 s por 5 min (métrica nativa do Cloud Run)
resource "google_monitoring_alert_policy" "latency" {
  display_name = "API de diagnostico - latencia p95 alta"
  combiner     = "OR"
  conditions {
    display_name = "p95 > 1s"
    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"diagnostico-api\" AND metric.type = \"run.googleapis.com/request_latencies\""
      comparison      = "COMPARISON_GT"
      threshold_value = 1000
      duration        = "300s"
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
        cross_series_reducer = "REDUCE_MAX"
      }
    }
  }
  notification_channels = var.alert_email == "" ? [] : [google_monitoring_notification_channel.email[0].id]
  depends_on            = [google_project_service.apis]
}
