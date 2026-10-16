CREATE VIEW v_content_latest AS
SELECT c.id AS content_id, c.platform, c.external_id, c.url, a.handle, a.relation, c.posted_at,
       m.observed_at, m.plays, m.likes, m.comments, m.shares
FROM content_item c
JOIN account a ON a.id = c.account_id
LEFT JOIN metric_snapshot m
  ON m.content_id = c.id
 AND m.observed_at = (SELECT max(observed_at) FROM metric_snapshot WHERE content_id = c.id);

CREATE VIEW v_content_growth AS
SELECT content_id, observed_at,
       plays - lag(plays) OVER w AS d_plays,
       likes - lag(likes) OVER w AS d_likes,
       (julianday(observed_at) - julianday(lag(observed_at) OVER w)) * 86400.0 AS d_seconds
FROM metric_snapshot
WINDOW w AS (PARTITION BY content_id ORDER BY observed_at);

CREATE VIEW v_trend_popularity AS
WITH ex AS (
  SELECT te.trend_id, l.plays,
         row_number() OVER (PARTITION BY te.trend_id ORDER BY l.plays) AS rn,
         count(*)     OVER (PARTITION BY te.trend_id) AS n
  FROM trend_example te
  JOIN v_content_latest l ON l.content_id = te.content_id
  WHERE l.plays IS NOT NULL
)
SELECT t.id AS trend_id, t.name, t.window_start, t.window_end,
       (SELECT count(*) FROM trend_example WHERE trend_id = t.id) AS example_count,
       (SELECT avg(plays) FROM ex WHERE ex.trend_id = t.id AND rn IN ((n + 1) / 2, (n + 2) / 2)) AS median_plays
FROM trend t;

CREATE VIEW v_project_overview AS
SELECT p.id AS project_id, p.title, p.kind, p.status, t.name AS primary_trend,
       (SELECT count(*) FROM project_reference r WHERE r.project_id = p.id) AS reference_count,
       (SELECT count(*) FROM project_output o WHERE o.project_id = p.id) AS output_count,
       (SELECT sum(l.plays) FROM project_output o JOIN v_content_latest l ON l.content_id = o.content_id
         WHERE o.project_id = p.id) AS output_plays
FROM project p
LEFT JOIN project_trend pt ON pt.project_id = p.id AND pt.role = 'primary'
LEFT JOIN trend t ON t.id = pt.trend_id;

CREATE VIEW v_post_vs_trend AS
SELECT po.project_id, po.content_id, l.relation, l.plays, pt.trend_id, tp.median_plays,
       CASE WHEN tp.median_plays > 0 THEN l.plays * 1.0 / tp.median_plays END AS ratio
FROM project_output po
JOIN v_content_latest l ON l.content_id = po.content_id
LEFT JOIN project_trend pt ON pt.project_id = po.project_id AND pt.role = 'primary'
LEFT JOIN v_trend_popularity tp ON tp.trend_id = pt.trend_id;

CREATE VIEW v_shot_lineage AS
SELECT sl.project_id, sl.id AS shotlist_id, sl.version, sc.shot_ord,
       sc.content_id, ci.url AS content_url, sc.web_source_id, ws.url AS source_url,
       st.id AS step_id, st.run_id
FROM shot_citation sc
JOIN shotlist sl ON sl.id = sc.shotlist_id
JOIN step st ON st.id = sl.created_by_step
LEFT JOIN content_item ci ON ci.id = sc.content_id
LEFT JOIN web_source ws ON ws.id = sc.web_source_id;

CREATE VIEW v_near_duplicates AS
SELECT a.blob_sha256 AS sha_a, a.frame_ms AS frame_a, b.blob_sha256 AS sha_b, b.frame_ms AS frame_b,
       hamming(a.phash, b.phash) AS distance
FROM media_phash a
JOIN media_phash b ON a.blob_sha256 < b.blob_sha256
WHERE hamming(a.phash, b.phash) <= 6;

CREATE VIEW v_current_steps AS
SELECT s.* FROM step s
WHERE NOT EXISTS (SELECT 1 FROM step n WHERE n.supersedes_step = s.id);

CREATE VIEW v_run_cost AS
SELECT s.run_id, s.stage, count(*) AS attempts, sum(s.cost_usd) AS cost_usd,
       (SELECT avg(lc.valid) FROM llm_call lc JOIN step s2 ON s2.id = lc.step_id
         WHERE s2.run_id = s.run_id AND s2.stage = s.stage) AS llm_valid_rate
FROM step s
GROUP BY s.run_id, s.stage;

CREATE VIEW v_media_style AS
SELECT ma.media_sha256, ma.analyzer_version, m.duration_ms,
       count(s.ord) AS shot_count,
       CASE WHEN m.duration_ms > 0 AND count(s.ord) > 0 THEN (count(s.ord) - 1) * 1000.0 / m.duration_ms END AS cuts_per_sec,
       ma.first_word_ms, ma.language
FROM media_analysis ma
JOIN media m ON m.blob_sha256 = ma.media_sha256
LEFT JOIN media_shot s ON s.media_sha256 = ma.media_sha256 AND s.analyzer_version = ma.analyzer_version
GROUP BY ma.media_sha256, ma.analyzer_version;
