# SQL and Data Modeling Study Notes

## Relational Thinking

Relational databases store data in tables with rows and columns. A primary key
uniquely identifies each row. A foreign key links one table to another.

Good schemas reduce duplication and make business rules clear.

## Query Building Blocks

The logical order of a SQL query is:

1. FROM and JOIN
2. WHERE
3. GROUP BY
4. HAVING
5. SELECT
6. ORDER BY
7. LIMIT

The written order is different, so debugging SQL is easier when you reason about
the logical order.

## Joins

An inner join keeps rows with matches in both tables. A left join keeps all rows
from the left table and fills missing matches with nulls.

Use joins carefully when one side has duplicate keys, because the result can
multiply rows and distort aggregates.

## Aggregation

Aggregates summarize rows:

- COUNT counts rows or non-null values.
- SUM adds numeric values.
- AVG gives a mean.
- MIN and MAX find extremes.

HAVING filters grouped results after aggregation. WHERE filters raw rows before
aggregation.

## Analytics Example

To evaluate model performance by segment:

```sql
SELECT
    customer_segment,
    COUNT(*) AS predictions,
    AVG(loss) AS avg_loss
FROM model_scoring
WHERE scored_at >= DATE '2026-01-01'
GROUP BY customer_segment
HAVING COUNT(*) >= 100
ORDER BY avg_loss DESC;
```

This query finds segments where the model performs worst while avoiding tiny
sample sizes.

