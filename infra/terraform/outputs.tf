output "api_url" {
  description = "URL pública da API (Swagger em /docs)"
  value       = google_cloud_run_v2_service.api.uri
}

output "image" {
  description = "Imagem esperada no Artifact Registry"
  value       = local.image
}
