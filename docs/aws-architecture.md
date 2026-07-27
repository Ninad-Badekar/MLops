# AWS architecture (portfolio)

## Components

| Service | Role |
|---------|------|
| S3 | DVC remote (`/dvc`), serving artifacts (`/serving`), MLflow artifacts (`/mlflow`) |
| ECR | Single image used by API and MLflow tasks |
| VPC | 2 public subnets, IGW, **no NAT** (tasks get public IPs) |
| ALB | `:80` → FastAPI, `:8080` → MLflow |
| ECS Fargate | `api` (0.25 vCPU / 512 MB), `mlflow` (same) |
| Secrets Manager | Generated `API_KEY` injected into the API task |
| CloudWatch Logs | `/ecs/...-api`, `/ecs/...-mlflow`; JSON prediction lines |
| IAM OIDC | GitHub Actions assumes deploy role (no long-lived keys) |

## Request path

1. Client → ALB:80 → ECS API → loads `MODEL_S3_URI` at startup
2. Each `/predict` emits a JSON log line (visible in CloudWatch) and updates an in-memory ring buffer for `/dashboard`
3. CI → `dvc repro` → `scripts/sync_serving_artifacts.py` → ECS force-new-deployment

## MLflow note

Backend store is SQLite on the **ephemeral** task filesystem. Artifact files live on S3. If the MLflow task is replaced, the run metadata DB resets (portfolio trade-off; no RDS/EFS cost).

## Estimated cost

Leaving the stack up 24/7 is typically **low tens of USD/month** in `ap-south-1`, driven mostly by the ALB. Mitigations:

- `desired_count = 0` in Terraform or via `aws ecs update-service`
- `terraform destroy` when not demoing
- S3 lifecycle expires old versions after 30 days / objects after 90 days

## Tear-down checklist

```bash
cd infra
terraform destroy
```

Also remove GitHub secrets if the role ARN will change on next apply.
