# Query Requirements and Expansion Guide

This guide explains:

1. what queries are needed in this repository,
2. how to get more queries, and
3. what high-quality queries should look like.

The codebase has two systems, so query needs are different for each.

---

## 1) Which queries are needed

## A. Legacy root classifier (`main.py`)

This system trains a fast/slow classifier from timed BIRD queries.

Required source format (BIRD JSON):

- Required fields: `db_id`, `SQL`
- Optional but used if present: `question_id`, `difficulty`
- Files expected: `mini_dev_sqlite.json` (preferred) or `mini_dev_mysql.json`
- Database files expected at: `dev_databases/<db_id>/<db_id>.sqlite`

After timing, the pipeline writes rows with:

- `question_id`, `db_id`, `sql`, `difficulty`, `runtime_s`

Then feature extraction and labels are built from SQL text.

## B. Runtime regression system (`sql_runtime_predictor/`)

This system predicts continuous runtime and expects synthetic query JSONL files.

Required synthetic query record shape:

```json
{
  "query_id": "synth_<db_name>_<index>",
  "query_text": "SELECT ...",
  "database": "<db_name>",
  "template_metadata": { "...": "..." }
}
```

Files are written under:

- `sql_runtime_predictor/data/synthetic_queries/<db>.jsonl`

Each query must be valid on that database and executable on SQLite.

---

## 2) What good queries should look like

Good query sets should cover both structure and runtime behavior.

## Structural diversity

Include a mix of:

- single-table and multi-table queries (1 to 4 tables),
- join styles (`INNER JOIN`, `LEFT JOIN`, and implicit cross-style),
- predicate counts from none to many (`WHERE` with `AND`/`OR`),
- numeric and text predicates (`=`, `<`, `>`, `BETWEEN`, `IN`, `LIKE`, `IS NULL`),
- aggregation patterns (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`) with and without `GROUP BY`/`HAVING`,
- result modifiers (`DISTINCT`, `ORDER BY`, `LIMIT`),
- advanced constructs (subqueries, occasional set ops such as `UNION`/`INTERSECT`/`EXCEPT`).

## Runtime diversity

Include queries that naturally span:

- very fast lookups,
- medium complexity analytics, and
- slower scans/joins/aggregations.

This runtime spread is important for both:

- binary labeling in the root pipeline (`fast`/`slow`), and
- stable regression behavior in `sql_runtime_predictor`.

## Validity requirements

Queries should:

- run successfully on SQLite for the target `db_id`,
- avoid dialect-only syntax unless converted first,
- return some rows when possible (not mandatory, but useful for realism),
- be reproducible and serializable in JSON/JSONL.

---

## 3) How to get more queries

## Option 1 (recommended): Generate more synthetic queries automatically

From `sql_runtime_predictor/`:

```bash
python -m src.creation.generate_queries --per-db 500
```

To scale up:

- increase `--per-db` directly, or
- raise defaults in `configs/default.yaml`:
  - `queries_per_db_min`
  - `queries_per_db_max`
  - `max_generation_attempts_per_db`

Generate for one database only:

```bash
python -m src.creation.generate_queries --db <db_id> --per-db 1200
```

## Option 2: Add manual query templates

You can add your own queries into `data/synthetic_queries/<db>.jsonl` using the same record format:

```json
{"query_id":"manual_financial_00001","query_text":"SELECT ...","database":"financial","template_metadata":{"source":"manual"}}
```

Best practice:

- keep `query_id` unique,
- keep SQL normalized (no trailing semicolon required),
- include metadata tags for later filtering.

## Option 3: Expand from BIRD JSON assets

For the root pipeline:

- ensure you have `mini_dev_sqlite.json` (or `mini_dev_mysql.json` with conversion),
- verify all matching SQLite DB files exist under `dev_databases`,
- run timing again to refresh `data/query_dataset_raw.csv`.

If only MySQL-style SQL is available, conversion is handled in code via `convert_mysql_to_sqlite(...)` before execution.

## Option 4: Iterate with runtime collection feedback

After generation:

1. collect runtimes (`src.creation.collect_runtimes`),
2. inspect timeout/error-heavy patterns,
3. adjust generation volume and template mix,
4. regenerate and retime.

This loop improves both quality and coverage.

---

## 4) Practical quality checklist before training

- Query files exist per database in `synthetic_queries/`
- JSONL schema is valid (`query_id`, `query_text`, `database`)
- Queries execute on SQLite without major failure rates
- Feature diversity exists (joins, predicates, aggregation, nesting)
- Runtime distribution is not collapsed into one narrow band
- No single database dominates the full training set

---

## 5) Example query shapes to include

Simple filter:

```sql
SELECT customer_id, balance
FROM accounts
WHERE balance > 1000
LIMIT 100
```

Join with predicates:

```sql
SELECT c.name, o.order_id, o.total
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.total BETWEEN 50 AND 500
ORDER BY o.total DESC
LIMIT 200
```

Aggregation with group and having:

```sql
SELECT o.customer_id, COUNT(*) AS n_orders, AVG(o.total) AS avg_total
FROM orders o
GROUP BY o.customer_id
HAVING AVG(o.total) > 100
ORDER BY avg_total DESC
```

Subquery:

```sql
SELECT p.product_id, p.price
FROM products p
WHERE p.price < (
  SELECT MAX(price)
  FROM products
)
```

Including these patterns helps the model see both straightforward and complex execution behavior.
