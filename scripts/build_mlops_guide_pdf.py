#!/usr/bin/env python3
"""Build a polished MLOps guide PDF for the Water Potability project."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "MLOps_Document_Style_Guide_v2.pdf"

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

# Palette
NAVY = colors.HexColor("#0B1F3A")
BLUE = colors.HexColor("#1D4ED8")
BLUE_SOFT = colors.HexColor("#EFF6FF")
SLATE = colors.HexColor("#64748B")
TEXT = colors.HexColor("#1E293B")
BORDER = colors.HexColor("#E2E8F0")
WHITE = colors.white
CYAN = colors.HexColor("#0E7490")
GREEN = colors.HexColor("#15803D")
AMBER = colors.HexColor("#B45309")


def build_styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="CoverTitle", fontName="Helvetica-Bold", fontSize=32, leading=38,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=10,
    ))
    base.add(ParagraphStyle(
        name="CoverSub", fontName="Helvetica", fontSize=14, leading=20,
        textColor=SLATE, alignment=TA_CENTER, spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        name="CoverTag", fontName="Helvetica", fontSize=10, leading=14,
        textColor=BLUE, alignment=TA_CENTER,
    ))
    base.add(ParagraphStyle(
        name="Section", fontName="Helvetica-Bold", fontSize=18, leading=22,
        textColor=NAVY, spaceBefore=4, spaceAfter=10,
    ))
    base.add(ParagraphStyle(
        name="SubSection", fontName="Helvetica-Bold", fontSize=12, leading=15,
        textColor=BLUE, spaceBefore=8, spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        name="Body", fontName="Helvetica", fontSize=10.5, leading=15,
        textColor=TEXT, alignment=TA_JUSTIFY, spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        name="DocBullet", fontName="Helvetica", fontSize=10.5, leading=15,
        textColor=TEXT, leftIndent=14, bulletIndent=4, spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        name="TH", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=WHITE,
    ))
    base.add(ParagraphStyle(
        name="TD", fontName="Helvetica", fontSize=9, leading=12.5, textColor=TEXT,
    ))
    base.add(ParagraphStyle(
        name="TDBold", fontName="Helvetica-Bold", fontSize=9, leading=12.5, textColor=TEXT,
    ))
    base.add(ParagraphStyle(
        name="FlowTitle", fontName="Helvetica-Bold", fontSize=9.5, leading=12,
        textColor=NAVY, alignment=TA_CENTER,
    ))
    base.add(ParagraphStyle(
        name="FlowSub", fontName="Helvetica", fontSize=8, leading=10,
        textColor=SLATE, alignment=TA_CENTER,
    ))
    base.add(ParagraphStyle(
        name="Arrow", fontName="Helvetica", fontSize=14, leading=14,
        textColor=BLUE, alignment=TA_CENTER,
    ))
    base.add(ParagraphStyle(
        name="TOC", fontName="Helvetica", fontSize=11, leading=18, textColor=TEXT,
    ))
    base.add(ParagraphStyle(
        name="Callout", fontName="Helvetica-Oblique", fontSize=10, leading=14,
        textColor=NAVY, backColor=BLUE_SOFT, borderPadding=10, spaceBefore=6, spaceAfter=6,
    ))
    return base


S = build_styles()


def draw_page_frame(canvas, doc, *, cover=False):
    canvas.saveState()
    if cover:
        # Top brand band
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 1.2 * cm, PAGE_W, 1.2 * cm, fill=1, stroke=0)
        canvas.setFillColor(BLUE)
        canvas.rect(0, PAGE_H - 1.2 * cm, PAGE_W, 0.18 * cm, fill=1, stroke=0)
        canvas.restoreState()
        return

    # Content pages: subtle header line + footer
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, PAGE_H - MARGIN + 4, PAGE_W - MARGIN, PAGE_H - MARGIN + 4)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE)
    canvas.drawString(MARGIN, 1.4 * cm, "Water Potability MLOps — Implementation Guide")
    canvas.drawRightString(PAGE_W - MARGIN, 1.4 * cm, f"Page {doc.page - 1}")

    # Left accent
    canvas.setFillColor(BLUE)
    canvas.rect(MARGIN - 0.55 * cm, 2.0 * cm, 0.12 * cm, PAGE_H - 4.0 * cm, fill=1, stroke=0)
    canvas.restoreState()


def make_table(rows, col_widths, header=True):
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, text in enumerate(row):
            if r == 0 and header:
                cells.append(Paragraph(text, S["TH"]))
            elif c == 0:
                cells.append(Paragraph(f"<b>{text}</b>", S["TDBold"]))
            else:
                cells.append(Paragraph(text, S["TD"]))
        data.append(cells)

    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F8FAFC")]),
        ]
    else:
        cmds.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, colors.HexColor("#F8FAFC")]))
    t.setStyle(TableStyle(cmds))
    return t


def flow_box(title, subtitle, bg, border):
    inner = Table(
        [[Paragraph(title, S["FlowTitle"])], [Paragraph(subtitle, S["FlowSub"])]],
        colWidths=[CONTENT_W - 1.6 * cm],
    )
    inner.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    outer = Table([[inner]], colWidths=[CONTENT_W])
    outer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 1.2, border),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return outer


def architecture_flow():
    """Reliable table-based vertical flow — no custom drawing overlap."""
    steps = [
        ("GitHub Repository", "Source code, tests, and DVC configuration", BLUE_SOFT, BLUE),
        ("GitHub Actions CI/CD", "flake8 · pytest · dvc repro", BLUE_SOFT, BLUE),
        ("DVC Pipeline", "Collect → preprocess → train → evaluate", colors.HexColor("#ECFEFF"), CYAN),
        ("MLflow Tracking", "Parameters, metrics, and model artifacts", colors.HexColor("#ECFEFF"), CYAN),
        ("deploy.zip", "Packaged FastAPI app and model.pkl", colors.HexColor("#FFF7ED"), AMBER),
        ("Azure App Service", "Python 3.12 inference API (uvicorn)", colors.HexColor("#F0FDF4"), GREEN),
        ("Monitoring Layer", "Latency, drift, and prediction statistics", colors.HexColor("#F0FDF4"), GREEN),
    ]
    rows = []
    for i, (title, sub, bg, border) in enumerate(steps):
        rows.append([flow_box(title, sub, bg, border)])
        if i < len(steps) - 1:
            rows.append([Paragraph("▼", S["Arrow"])])
    t = Table(rows, colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    return t


def concept_grid():
    items = [
        ("Reproducibility", "DVC pipeline + params.yaml"),
        ("Experiment Tracking", "MLflow run logging"),
        ("Automation", "GitHub Actions CI/CD"),
        ("Monitoring", "Drift and latency API"),
    ]
    row1 = []
    row2 = []
    for i, (t, s) in enumerate(items):
        cell = Table(
            [[Paragraph(f"<b>{t}</b>", S["TDBold"])], [Paragraph(s, S["TD"])]],
            colWidths=[(CONTENT_W - 0.4 * cm) / 2],
        )
        cell.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BLUE_SOFT),
            ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        (row1 if i < 2 else row2).append(cell)

    t = Table([row1, row2], colWidths=[(CONTENT_W - 0.4 * cm) / 2] * 2, rowHeights=[None, None])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def build_story():
    col1 = 4.2 * cm
    col2 = CONTENT_W - col1
    story = []

    # ── Cover ──────────────────────────────────────────────────────────
    story += [
        Spacer(1, 3.5 * cm),
        Paragraph("MLOps Concepts &<br/>Project Implementation", S["CoverTitle"]),
        Spacer(1, 0.4 * cm),
        Paragraph("Water Potability Prediction System", S["CoverSub"]),
        Spacer(1, 0.8 * cm),
        Paragraph(
            "A clear, practical guide explaining what MLOps is,<br/>"
            "how this project implements it, and how the system is architected.",
            S["CoverSub"],
        ),
        Spacer(1, 1.2 * cm),
        Paragraph("DVC  ·  MLflow  ·  FastAPI  ·  GitHub Actions  ·  Terraform  ·  Azure", S["CoverTag"]),
        Spacer(1, 4 * cm),
        Paragraph("Document Edition · v2", S["CoverTag"]),
        PageBreak(),
    ]

    # ── TOC ────────────────────────────────────────────────────────────
    story += [
        Paragraph("Contents", S["Section"]),
        Spacer(1, 6),
    ]
    for n, title in [
        ("1", "What is MLOps?"),
        ("2", "Core MLOps Concepts"),
        ("3", "Project Overview"),
        ("4", "Architecture Diagram"),
        ("5", "Concept-to-Implementation Mapping"),
        ("6", "Roadmap & Next Steps"),
    ]:
        story.append(Paragraph(f"<b>{n}.</b>&nbsp;&nbsp;{title}", S["TOC"]))
    story.append(PageBreak())

    # ── 1. What is MLOps ───────────────────────────────────────────────
    story += [
        Paragraph("1. What is MLOps?", S["Section"]),
        Paragraph(
            "MLOps (Machine Learning Operations) is the discipline of reliably building, deploying, "
            "and operating machine learning systems in production. It brings together data science, "
            "software engineering, and operations — extending DevOps to cover data and models, not just code.",
            S["Body"],
        ),
    ]
    for b in [
        "Makes training and deployment <b>reproducible</b> from tracked data and parameters.",
        "Adds <b>quality gates</b> before models reach production.",
        "Enables <b>monitoring</b> of latency, errors, and data drift after release.",
        "Closes the gap between notebook experiments and production ML services.",
    ]:
        story.append(Paragraph(b, S["DocBullet"], bulletText="•"))

    story += [
        Spacer(1, 6),
        Paragraph(
            "Without MLOps, teams rely on manual steps, lose experiment history, and struggle to "
            "detect when models degrade in production.",
            S["Callout"],
        ),
        Spacer(1, 12),
    ]

    # ── 2. Core Concepts ───────────────────────────────────────────────
    story += [
        Paragraph("2. Core MLOps Concepts", S["Section"]),
        make_table([
            ["Concept", "What it means"],
            ["Data Versioning", "Track which dataset version trained each model."],
            ["Experiment Tracking", "Log parameters, metrics, and artifacts across runs."],
            ["Pipeline Orchestration", "Automate ordered steps from data prep to evaluation."],
            ["CI/CD for ML", "Run tests, training checks, and deployment automatically."],
            ["Model Serving", "Expose the model as a reliable API for end users."],
            ["Monitoring & Drift", "Detect latency issues and distribution shifts in production."],
            ["Governance", "Control promotion, approval, rollback, and auditability."],
        ], [col1, col2]),
        PageBreak(),
    ]

    # ── 3. Project Overview ────────────────────────────────────────────
    story += [
        Paragraph("3. Project Overview", S["Section"]),
        Paragraph(
            "This project predicts whether water is <b>potable</b> using nine chemical features. "
            "It demonstrates a complete MLOps lifecycle deployed on <b>Azure for Students Starter</b> "
            "at zero cost.",
            S["Body"],
        ),
        Spacer(1, 8),
        concept_grid(),
        Spacer(1, 12),
        make_table([
            ["Component", "Role in this project"],
            ["FastAPI", "Serves predictions, health checks, and a monitoring dashboard."],
            ["DVC", "Defines reproducible data and training pipeline stages."],
            ["MLflow", "Logs experiment parameters, metrics, and model artifacts."],
            ["GitHub Actions", "Runs linting, tests, training, packaging, and deployment."],
            ["Terraform", "Provisions Azure infrastructure as code."],
        ], [col1, col2]),
        PageBreak(),
    ]

    # ── 4. Architecture ────────────────────────────────────────────────
    story += [
        Paragraph("4. Architecture Diagram", S["Section"]),
        Paragraph(
            "The system follows a vertical flow from code commit to production monitoring. "
            "Each stage is automated and auditable.",
            S["Body"],
        ),
        Spacer(1, 8),
        architecture_flow(),
        Spacer(1, 10),
        Paragraph(
            "<b>Request path:</b> Client → POST /predict → FastAPI → model inference → "
            "response + monitoring logs",
            S["Callout"],
        ),
        PageBreak(),
    ]

    # ── 5. Mapping ─────────────────────────────────────────────────────
    story += [
        Paragraph("5. Concept-to-Implementation Mapping", S["Section"]),
        make_table([
            ["MLOps Concept", "How we implemented it"],
            ["Reproducibility", "DVC stages in dvc.yaml and hyperparameters in params.yaml."],
            ["Experiment Tracking", "MLflow logs runs and registers WaterPotabilityModel."],
            ["Automation", "GitHub Actions runs flake8, pytest, dvc repro, and deploys."],
            ["Deployment", "deploy.zip is released to Azure App Service."],
            ["Monitoring", "API tracks latency, class distribution, and feature drift."],
        ], [col1, col2]),
        Spacer(1, 16),
    ]

    # ── 6. Roadmap ───────────────────────────────────────────────────
    story += [
        Paragraph("6. Roadmap & Next Steps", S["Section"]),
    ]
    for b in [
        "Add model promotion stages: development → staging → production.",
        "Define drift alerting thresholds and notification workflows.",
        "Implement automated retraining when quality degrades.",
        "Adopt managed artifact storage and an MLflow tracking server.",
        "Introduce canary or blue-green deployment for safer rollouts.",
    ]:
        story.append(Paragraph(b, S["DocBullet"], bulletText="•"))

    story += [
        Spacer(1, 12),
        Paragraph(
            "<b>Key takeaway:</b> This project demonstrates the full ML lifecycle — not just model "
            "training. It is a practical foundation that can evolve into enterprise-grade MLOps.",
            S["Callout"],
        ),
    ]

    return story


def main():
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="MLOps Concepts and Project Implementation Guide",
        author="Water Potability MLOps Project",
    )

    def on_page(canvas, doc):
        draw_page_frame(canvas, doc, cover=(doc.page == 1))

    doc.build(build_story(), onFirstPage=on_page, onLaterPages=on_page)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
