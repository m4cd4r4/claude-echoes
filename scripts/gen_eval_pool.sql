-- Candidate pool for the generated retrieval eval.
--
-- Two rules keep this honest:
--   1. Questions are FILTERED ON THE QUESTION ALONE. Nothing here looks at the
--      gold answer, so the set cannot be biased toward what retrieval can serve.
--   2. Gold is STRUCTURAL - the next substantive assistant turn in the same
--      session. No editorial judgement of relevance is made anywhere.
--
-- The ordering is md5(id) rather than random() so the sample is reproducible.
WITH q AS (
  SELECT id, session_id, project, created_at,
         btrim(regexp_replace(content, E'[\r\n\t]+', ' ', 'g')) AS content
  FROM messages
  WHERE role = 'user'
    AND source = 'chat'
    AND content LIKE '%?%'
    AND length(content) BETWEEN 45 AND 260
    AND content NOT LIKE '%task-notification%'
    AND content NOT LIKE '%<system-reminder>%'
    AND content NOT LIKE '%<ide_%'
    AND content NOT LIKE '%<local-command%'
    AND content NOT LIKE '%<command-name>%'
    AND content !~* '^(yes|no|ok|okay|sure|continue|proceed|please proceed|go ahead|thanks|thank you|so[ ,]|but |and |also )'
    AND content !~* '(screenshot|see above|as above|this one|these |those |that one|the above|attached)'
    AND array_length(regexp_split_to_array(btrim(content), '\s+'), 1) >= 8
    AND created_at < NOW() - INTERVAL '2 days'
),
g AS (
  SELECT q.id AS qid, q.session_id, q.project, q.created_at AS asked_at, q.content AS question,
         (SELECT m.id FROM messages m
           WHERE m.session_id = q.session_id AND m.role = 'assistant' AND m.source = 'chat'
             AND m.created_at > q.created_at AND length(m.content) >= 200
           ORDER BY m.created_at ASC LIMIT 1) AS gold_id
  FROM q
)
SELECT row_to_json(t) FROM (
  SELECT g.qid, g.session_id, g.project, g.asked_at, g.question, g.gold_id,
         left(btrim(regexp_replace(m.content, E'[\r\n\t]+', ' ', 'g')), 300) AS gold_snippet
  FROM g JOIN messages m ON m.id = g.gold_id
  WHERE g.gold_id IS NOT NULL
  ORDER BY md5(g.qid::text)
  LIMIT 400
) t;
