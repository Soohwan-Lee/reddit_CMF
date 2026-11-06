# Ethics and Data Usage Guidelines

- Only publicly available Reddit data is collected.
- Direct identifiers are excluded; author names are pseudonymised (`u/[anon-xxxxxxxx]`, a stable SHA-256 hash, optionally salted via `ANON_SALT`).
- Sensitive images and comments are excluded.
- The goal is to understand the phenomenon and contextualise design practice, not to criticise the community or individuals.
- When quoting, anonymisation rules are followed.
- If the dataset is ever released, only an anonymised version is published.

In this repository, raw and processed data, embeddings, and the interactive HTML plot (which shows post titles) are gitignored. Only aggregate figures are committed, and the README reports cluster-level themes and statistics rather than individual posts.
