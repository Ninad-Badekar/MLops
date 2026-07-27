# Water Potability MLOps (Azure Students Starter)

End-to-end MLOps demo that **runs on Azure for Students Starter**: DVC + MLflow in CI, FastAPI on **App Service (Free F1)**.

> Starter blocks Storage, ACR, Key Vault, Container Apps, etc. This repo is tailored to what Starter allows (`Microsoft.Web`, `Microsoft.Insights`).

## Architecture

```text
GitHub Actions
  ├─ dvc repro + pytest + MLflow (sqlite)
  └─ deploy.zip (API + model.pkl) ──► Azure App Service (Python 3.12)
                                         └─ Application Insights
```

See [docs/azure-architecture.md](docs/azure-architecture.md).

## Local quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc repro
uvicorn src.main:app --reload --port 8000
```

- API: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  
- Dashboard: http://127.0.0.1:8000/dashboard  

## Deploy on Students Starter

### 1. Login

```bash
az login
az account show
```

### 2. Clean previous failed Terraform state (important)

If you already tried the full Container Apps stack:

```bash
cd ~/MLops/MLops/infra
export PATH="$HOME/.local/bin:$PATH"
terraform destroy -auto-approve || true
rm -rf .terraform terraform.tfstate terraform.tfstate.backup
```

### 3. Apply App Service infra

```bash
cd ~/MLops/MLops/infra
cp terraform.tfvars.example terraform.tfvars   # already filled if present
# ensure subscription_id matches az account show
terraform init -upgrade
terraform apply
```

Type `yes`. This creates only: Resource Group, Free App Service Plan, Linux Web App, Application Insights.

### 4. Save outputs

```bash
terraform output
terraform output -raw api_key
terraform output -raw get_publish_profile_command
# run the printed az command, copy XML:
az webapp deployment list-publishing-profiles \
  --name "$(terraform output -raw webapp_name)" \
  --resource-group "$(terraform output -raw resource_group)" \
  --xml
```

### 5. GitHub secrets

| Secret | Value |
|--------|--------|
| `AZURE_WEBAPP_NAME` | `terraform output -raw webapp_name` |
| `AZURE_WEBAPP_PUBLISH_PROFILE` | full XML from the `az ... --xml` command |

### 6. Push to main

Push/merge to `main`. CI trains, zips the API + model, deploys to App Service.

Open: `terraform output -raw api_url`

Predict:

```bash
API_URL=$(cd infra && terraform output -raw api_url)
API_KEY=$(cd infra && terraform output -raw api_key)
curl -X POST "$API_URL/predict" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"ph":7,"Hardness":200,"Solids":20000,"Chloramines":7,"Sulfate":300,"Conductivity":400,"Organic_carbon":15,"Trihalomethanes":60,"Turbidity":4}'
```

## Tear down

```bash
cd infra && terraform destroy
```

## License

MIT — see [LICENSE](LICENSE).
