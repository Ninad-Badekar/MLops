terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

resource "random_string" "suffix" {
  length  = 6
  upper   = false
  special = false
}

resource "random_password" "api_key" {
  length  = 32
  special = false
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  suffix      = random_string.suffix.result
  # App Service name: 2-60 chars, globally unique
  webapp_name = "wp-api-${local.suffix}"
}

resource "azurerm_resource_group" "main" {
  name     = "${local.name_prefix}-rg"
  location = var.azure_location

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Note        = "Azure-for-Students-Starter-compatible"
  }
}

# Free F1 works with code deploy (zip). Custom containers need paid SKU.
resource "azurerm_service_plan" "main" {
  name                = "wp-plan-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = var.app_service_sku
}

resource "azurerm_linux_web_app" "api" {
  name                = local.webapp_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  service_plan_id     = azurerm_service_plan.main.id
  https_only          = true

  site_config {
    always_on = var.app_service_sku != "F1"

    application_stack {
      python_version = "3.12"
    }

    app_command_line = "uvicorn src.main:app --host 0.0.0.0 --port 8000"
  }

  app_settings = {
    API_KEY                        = random_password.api_key.result
    MLFLOW_TRACKING_URI            = "sqlite:///mlflow.db"
    SCM_DO_BUILD_DURING_DEPLOYMENT = "true"
    WEBSITES_PORT                  = "8000"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
