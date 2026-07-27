# Azure architecture — Students Starter edition

This portfolio deploy targets **Azure for Students Starter**, which only allows a small set of providers (notably `Microsoft.Web` and `Microsoft.Insights`).

## What we use

| Service | Role |
|---------|------|
| Resource Group | Single tear-down unit |
| App Service Plan (F1 Free) | Hosts the Linux Python Web App |
| Linux Web App | FastAPI API + dashboard; model shipped inside the deploy zip |
| Application Insights | Basic telemetry (`Microsoft.Insights`) |

## What we intentionally skip (blocked on Starter)

- Blob Storage / DVC Azure remote
- ACR / Container Apps
- Key Vault / Managed Identity
- Log Analytics / Azure AD OIDC service principals

## Request path

1. GitHub Actions runs `dvc repro` (train locally in CI)
2. Packages `requirements-serve.txt` + `src/` + `models/model.pkl` into `deploy.zip`
3. Deploys zip to App Service via **publish profile**
4. App Service builds deps with Oryx and starts `uvicorn`

## Cost

F1 Free tier: **$0** (quota-limited). Tear down with `terraform destroy` when done.

## Limitations (honest)

- Cold starts / sleep on Free tier
- No cloud MLflow UI (tracking stays `sqlite` in CI)
- No shared artifact store — model is baked into each deploy
- Scale / always-on requires upgrading SKU (e.g. B1) or subscription
