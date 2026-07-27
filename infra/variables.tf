variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "azure_location" {
  description = "Azure region for all resources"
  type        = string
  default     = "centralindia"
}

variable "project_name" {
  description = "Short project name used in resource names"
  type        = string
  default     = "water-potability-mlops"
}

variable "environment" {
  description = "Environment label"
  type        = string
  default     = "portfolio"
}

variable "app_service_sku" {
  description = "App Service plan SKU. F1 = Free (Students Starter). Use B1 if you need always-on."
  type        = string
  default     = "F1"
}
