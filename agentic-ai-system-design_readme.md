# 🤖 Agentic AI — Partner Funding Intelligence System

## Overview
Designed and built a multi-source AI agent that combines qualitative knowledge (program documentation, wikis, guides) with quantitative data analysis (SQL-backed metrics) to serve as a unified intelligence layer for a cloud provider's Partner Funding programs.

The agent handles natural language queries across 8 funding programs, triages stuck requests, and provides data-driven answers to operational and strategic questions.

---

## Problem Statement
Partner funding operations teams were spending significant time:
- Manually looking up request statuses across multiple systems
- Answering repetitive eligibility and process questions
- Running ad-hoc SQL queries for metrics and reporting
- Navigating multiple dashboards and wikis for information

**Goal**: Build a single conversational interface that unifies all funding knowledge and data access.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Query                         │
│   "How many requests were approved for Program A    │
│    in Q1 2026 in NAMER?"                            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Multi-Agent Orchestrator                 │
│   Routes to best agent(s) based on query intent      │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  Quantitative │ │Qualitative│ │  Dashboard   │
│  Data Agent   │ │ Knowledge │ │  Integration │
│              │ │   Agent   │ │              │
│ SQL queries  │ │ Wiki/PDF  │ │  QuickSight  │
│ against      │ │ program   │ │  visual      │
│ metrics DB   │ │ rules &   │ │  reporting   │
│              │ │ processes │ │              │
└──────────────┘ └──────────┘ └──────────────┘
```

---

## Data Sources Integrated

| Source | Type | What It Provides |
|--------|------|-----------------|
| Metrics Super Table | Structured (SQL) | Request metrics, approved amounts, launched ARR, partner/geo/program breakdowns |
| Funding Dashboard | Visual (QuickSight) | Business reporting, wallet views, request tracking |
| Program Wiki | Unstructured (text) | Program eligibility, submission workflows, deadlines, escalation paths |
| Benefits Guide | Document (PDF) | Benefit amounts by designation, program caps, reimbursement rules |

---

## Capabilities

### Quantitative Queries
- Request metrics by program, geo, partner, time period
- Approved amounts and launched ARR analysis
- Stuck/orphaned request identification and triage
- Aging reports and pipeline health metrics
- YoY comparisons and trend analysis

### Qualitative Assistance
- Program eligibility determination across 8 programs
- Submission workflow guidance
- Deadline and expiration rule lookups
- Escalation path routing
- Payment platform troubleshooting (including migration support)

### Example Queries the Agent Handles
```
"Total partners that submitted requests for Program A in June 2025?"
"What are the eligibility requirements for Program B?"
"Look up request #12345 — why is it stuck?"
"YoY change in approved amounts for Program C?"
"How do I escalate a delayed reimbursement?"
"Total launched opportunities for 2025 tied to Program D?"
```

---

## Technical Implementation

### Agent Design Principles
| Principle | Implementation |
|-----------|---------------|
| Multi-agent routing | Orchestrator classifies query intent and routes to specialized agent |
| Row-Level Security | Data access filtered per user's authorization level |
| Source attribution | Every answer cites which source (DB, wiki, PDF) it came from |
| Graceful fallback | If one source fails, agent explains limitation and suggests alternatives |
| Session continuity | Follow-up questions maintain context without re-specifying parameters |

### Security & Access Control
- Row-Level Security (RLS) enforced on all data queries
- Users only see data for their authorized partners/geos/programs
- No raw SQL exposed to end users — agent generates and executes queries internally

---

## Impact

| Metric | Before | After |
|--------|--------|-------|
| Time to answer funding questions | 15-30 min (manual lookup) | < 30 seconds |
| Systems consulted per query | 3-4 (wiki + dashboard + DB + PDF) | 1 (agent) |
| Escalation routing accuracy | Variable (depends on who you ask) | Consistent (rules-based) |
| Self-service capability | Low (required analyst support) | High (natural language) |

---

## Predictive Model Extension (Proposed)

Beyond the operational agent, I proposed a predictive ML layer to forecast ARR from funded opportunities:

### Model Design
| Component | Detail |
|-----------|--------|
| Model | XGBoost Regressor |
| Target | Launched ARR per funded opportunity |
| Key Features | Program type, funding amount, partner tier, partner historical ROI, customer geo/segment |
| Evaluation | MAPE |
| Transformation | Log transform (ARR is right-skewed) |

### Partner Performance Framework
```
Performance Score = Actual ARR / Predicted ARR

> 1.3  → Overperformer (increase funding)
1.0-1.3 → On Track (maintain)
0.7-1.0 → Underperformer (review & coach)
< 0.7  → Significant Underperformer (reduce funding)
```

### Why This Matters
Raw ARR comparisons are unfair — a partner generating $400K ARR in APJ SMB with $50K funding may be outperforming a partner generating $2M in NAMER Enterprise with $500K funding. The model adjusts for context.

---

## Skills Demonstrated

| Skill | Application |
|-------|-------------|
| Agentic AI Design | Multi-agent orchestration, intent routing, source integration |
| Data Engineering | Connecting structured (SQL) + unstructured (wiki/PDF) sources |
| RAG Architecture | Retrieval-augmented generation for document-based Q&A |
| ML Proposal & Business Case Writing | Formal business case with stakeholder value, risks, timeline |
| Business Acumen | Translating operational pain points into technical solutions |
| Security Design | Row-level security, access control, data governance |

---

## Tools & Technologies
- Multi-Agent Orchestrator (MCP-based)
- SQL (metrics database)
- QuickSight (dashboard integration)
- Document parsing (PDF/wiki ingestion)
- XGBoost (proposed predictive model)
- Python, pandas, scikit-learn

---

## Files
```
├── README.md
├── docs/
│   ├── ARR_Prediction_Model_Proposal.md   # Full ML proposal
│   └── Agent_Architecture.md              # Design decisions & tradeoffs
```

---

## Note
This project was built within an enterprise cloud provider's internal systems. Code and data cannot be shared publicly due to confidentiality. This README documents the architecture, approach, and impact for portfolio purposes.
