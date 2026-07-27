output "resource_group" {
  value = azurerm_resource_group.main.name
}

output "azure_location" {
  value = var.azure_location
}

output "app_service_plan" {
  value = azurerm_service_plan.main.name
}

output "webapp_name" {
  description = "Set as AZURE_WEBAPP_NAME GitHub secret"
  value       = azurerm_linux_web_app.api.name
}

output "api_url" {
  value = "https://${azurerm_linux_web_app.api.default_hostname}"
}

output "api_key" {
  description = "X-API-Key for /predict, /api/simulate, /api/clear-logs"
  value       = random_password.api_key.result
  sensitive   = true
}

output "get_publish_profile_command" {
  description = "Run this to get AZURE_WEBAPP_PUBLISH_PROFILE for GitHub Actions"
  value       = "az webapp deployment list-publishing-profiles --name ${azurerm_linux_web_app.api.name} --resource-group ${azurerm_resource_group.main.name} --xml"
}
