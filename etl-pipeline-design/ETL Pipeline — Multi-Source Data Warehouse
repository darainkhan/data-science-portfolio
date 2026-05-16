# 🔄 ETL Pipeline — Multi-Source Data Warehouse

> **Disclaimer**: All naming conventions, table names, column names, program names, and business logic details have been anonymized or altered due to confidentiality. This document showcases the architectural design, technical patterns, and engineering decisions only.

---

## Overview
Designed and built a production ETL pipeline that consolidates operational data from 3 disparate source systems into a unified analytical layer. The pipeline handles multi-source ingestion, workflow state tracking, currency normalization, billing reconciliation, deduplication, and exports to a data lake for downstream analytics and AI consumption.

**Scale**: 200+ columns, 500K+ records, 15+ source tables, daily refresh

---

## Problem Statement
Operational data lived across 3 separate systems with:
- Different schemas, naming conventions, and ID formats
- No unified view for cross-system reporting
- Monetary amounts in local currencies without USD normalization
- Duplicate records from overlapping data sources
- No linkage between requests, issued credits, and actual billing usage

**Goal**: Build a single source of truth that unifies all operational data for reporting, analytics, and AI-powered querying.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SOURCE SYSTEMS                             │
├──────────────────┬──────────────────┬───────────────────────────┤
│  System A        │  System B        │  System C                  │
│  (Internal App)  │  (CRM Platform)  │  (Approval Platform)       │
│                  │                  │                            │
│  • Requests      │  • Requests      │  • Approvals               │
│  • Claims        │  • Claims        │  • PO Numbers              │
│  • Credits       │  • Projects      │  • Budget Tracking         │
│  • Workloads     │  • Opportunities │  • Vendor Metadata         │
│  • Attachments   │  • Accounts      │                            │
└────────┬─────────┴────────┬─────────┴──────────────┬────────────┘
         │                  │                        │
         ▼                  ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: EXTRACTION                            │
│                                                                  │
│  • Workflow history pivot (stage timestamps)                      │
│  • Record deduplication (ROW_NUMBER partitioning)                 │
│  • Vendor/account attribute enrichment                           │
│  • Cost center lookups                                           │
│  • Exchange rate joins for currency conversion                   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 2: CONSOLIDATION                         │
│                                                                  │
│  • System A consolidated (requests + claims + credits +          │
│    workloads + attachments + currency conversion)                 │
│  • System B consolidated (CRM records + projects +               │
│    claims + opportunity linkage)                                  │
│  • System C consolidated (approvals + program mapping +          │
│    PO extraction + amount cleaning)                              │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3: SOURCE OF TRUTH (SOT)                 │
│                                                                  │
│  UNION ALL of 3 consolidated sources with:                       │
│  • flag_datasource identifier per record                         │
│  • COALESCE logic for unified column naming                      │
│  • Engagement/project linkage                                    │
│  • Opportunity enrichment (customer, geo, segment)               │
│  • Vendor metadata (tier, type, geo, agreement status)           │
│  • Deduplication (ROW_NUMBER per request/claim/code)             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 4: STAGING                               │
│                                                                  │
│  • Joins SOT with opportunity data (2 opportunity systems)       │
│  • Enriches with vendor scorecard, account metadata              │
│  • Calculates approved amounts (cash + credit + invoiced)        │
│  • Applies business rules (exclusion lists, status filters)      │
│  • Exports to S3 as Parquet for data lake consumption            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 5: REPORTING                             │
│                                                                  │
│  • Super Table (final denormalized view)                         │
│  • Credits fact table (with billing usage reconciliation)        │
│  • Unified program naming, threshold-based amount logic          │
│  • Consumed by: Dashboards, AI Agent, Ad-hoc Analytics           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Components (4 SQL Scripts)

### Script 1: Source of Truth (SOT)
**Purpose**: Consolidate 3 source systems into a single unified table

| Challenge | Solution |
|-----------|----------|
| Different ID formats across systems | COALESCE with fallback logic |
| Different column names for same concept | Unified aliasing |
| Overlapping records | ROW_NUMBER partitioning for deduplication |
| Missing opportunity linkage | Multi-join strategy (primary → secondary → fallback) |
| Currency in local amounts | Exchange rate joins on transaction date |
| Workflow stage tracking | History pivot (CASE + MAX aggregation) |

**Key techniques**:
- 10+ LEFT JOINs for enrichment
- COALESCE chains (3-4 sources deep) for unified fields
- ROW_NUMBER OVER (PARTITION BY record_id) for dedup
- Currency conversion with date-matched exchange rates
- LISTAGG for multi-value fields
- S3 UNLOAD as Parquet with parallel export

---

### Script 2: Credits Fact Table
**Purpose**: Track promotional credit lifecycle from issuance through redemption and billing usage

| Stage | What's Tracked |
|-------|---------------|
| Issuance | Credit created, amount, vendor, program classification |
| Provisioning | Code provisioned, expiration date |
| Redemption | Code redeemed by customer account |
| Billing Usage | Monthly credit burn from billing system |

**Key techniques**:
- CTE-based pipeline (5 CTEs → final SELECT)
- Program classification engine (40+ CASE WHEN rules)
- Billing reconciliation (joining credit codes to billing line items)
- Vendor agreement enrichment
- Year-based logic split (legacy vs new classification rules)
- ROW_NUMBER for code-level deduplication

---

### Script 3: Staging Layer
**Purpose**: Enrich SOT with opportunity, customer, and vendor metadata

| Enrichment | Source |
|------------|--------|
| Opportunity details | 2 opportunity systems (internal + CRM) |
| Customer attributes | Account hierarchy (geo, segment, industry, territory) |
| Vendor metadata | Scorecard, tier, type, program membership |
| Agreements | Strategic agreement status and pricing type |
| Cost centers | PO-to-cost-center mapping |

**Key techniques**:
- Multi-system opportunity resolution (COALESCE across 4 sources)
- Vendor scorecard join with ID parsing (SPLIT_PART)
- Business rule filtering (exclusion of test accounts)
- Conditional amount calculations with status + type + dedup guards
- S3 UNLOAD for data lake export

---

### Script 4: Extraction Layer
**Purpose**: Extract and clean raw data from each source system

| Source | Extraction Logic |
|--------|-----------------|
| System A | Requests + Claims + Credits + Workloads + Attachments joined, currency converted, approval history pivoted |
| System B | CRM records + projects + claims, opportunity link parsing from URL fields |
| System C | Approval records with amount cleaning (handle NaN, None, scientific notation, commas), PO extraction from tags, program name standardization |

**Key techniques**:
- Workflow history pivot (row-level events → columnar timestamps via CASE + MAX)
- URL parsing for opportunity IDs (SPLIT_PART, REPLACE, LEFT)
- Robust amount cleaning (REGEXP_REPLACE, NULLIF, multiple edge cases)
- PO number extraction from free-text tags field
- Date normalization across inconsistent formats
- Engagement ID resolution (multiple format variations)

---

## Technical Highlights

### Multi-Source Deduplication Strategy
```sql
ROW_NUMBER() OVER (PARTITION BY record_id) AS duplicate_rank
-- Only process rank = 1 for amount calculations
-- Prevents double-counting when same record exists in multiple systems
```

### Currency Conversion Pattern
```sql
CASE
    WHEN currency = 'USD' THEN local_amount
    ELSE local_amount * exchange_rate  -- Joined on transaction_date
END AS amount_usd
```

### Conditional Amount Calculation
```sql
-- Only count approved amounts when:
-- 1. Correct data source
-- 2. Correct status (approved/completed)
-- 3. Correct type (cash vs credit)
-- 4. Not a duplicate record (rank = 1)
-- 5. Not in exclusion list (extensions, wallet requests, etc.)
```

### Program Classification Engine
```sql
-- 40+ CASE WHEN rules mapping raw codes to standardized program names
-- Year-based logic split (pre-2025 vs post-2025 rules)
-- Handles naming variations, legacy codes, and edge cases
```

### Workflow History Pivot
```sql
-- Converts row-level audit events into columnar timestamps:
-- Input:  record_id | stage          | timestamp
--         FR-001    | created        | 2025-01-01
--         FR-001    | approved       | 2025-01-05
--         FR-001    | completed      | 2025-01-20
--
-- Output: record_id | created_date | approved_date | completed_date
--         FR-001    | 2025-01-01   | 2025-01-05    | 2025-01-20
```

---

## Data Quality Measures

| Issue | Handling |
|-------|----------|
| Duplicate records | ROW_NUMBER partitioning, process only rank 1 |
| NULL amounts | COALESCE with 0, NVL fallbacks |
| Dirty numeric fields | REGEXP_REPLACE, NULLIF for NaN/None/scientific notation |
| Inconsistent dates | TO_DATE with truncation, NULL handling |
| Test/internal accounts | WHERE exclusion list |
| Soft-deleted records | delete_flag filter on all CRM joins |
| Orphaned credits | Inner join to valid codes only |

---

## Output & Consumers

| Consumer | How They Use It |
|----------|----------------|
| BI Dashboards | Direct query for visual reporting |
| AI Agent | Natural language queries against unified table |
| Finance Team | Budget tracking, PO reconciliation |
| Account Managers | Vendor performance, utilization metrics |
| Leadership | Program-level metrics, geo breakdowns |

---

## Skills Demonstrated

| Skill | Application |
|-------|-------------|
| Data Warehousing | Layered architecture, fact + dimension design |
| ETL Design | Multi-layer pipeline (extract → consolidate → stage → report) |
| SQL (Advanced) | CTEs, window functions, COALESCE chains, CASE logic, LISTAGG |
| Data Quality | Deduplication, NULL handling, dirty data cleaning |
| Multi-Source Integration | 3 systems with different schemas unified |
| Currency Normalization | Date-matched exchange rate conversion |
| Data Lake Export | Parquet format, parallel S3 unload |
| Business Logic Translation | Complex rules encoded as SQL |

---

## Technologies
- Amazon Redshift (SQL engine)
- Amazon S3 (data lake storage, Parquet format)
- IAM (role-based access for data export)
- Scheduled execution (daily refresh)

---

## Note
This pipeline was built within an enterprise environment. All table names, column names, program names, business logic specifics, and identifiers have been changed or anonymized. This document showcases the architectural design and engineering patterns only — not the actual business domain or implementation details.
