# How I plan the next work (Scrum-style)

This is the plan for the next sprints of **energiewende-bundestag-nlp**. It is a planned way of working for a one-person project, not a record of past team work.

## Working agreement
- Sprint: one week. Sprint goal: one backlog item finished, with its numbers in `reports/` and the README.
- Planning on Monday (pick items), review on Friday (open the project page, check the numbers), a short retrospective note in the commit log.
- Board: one GitHub issue per backlog item, columns To do, In progress, Done (GitHub Projects).
- Branches: `main` stays working; each item gets its own branch and a pull request.

## Definition of done
1. The script runs in a fresh virtual environment from `requirements.txt`.
2. Every number in the README and on the page matches `reports/`.
3. The project page opens without console errors and works on a phone.
4. No secrets or personal data in the files or history.
5. Limits are stated next to the result.

## Product backlog
| # | User story | Acceptance |
|---|---|---|
| 1 | As a reader, I want LDA and BERTopic inside the script, so that all numbers are reproducible. | Coherence scores reproduced by a script. |
| 2 | As a reader, I want German-language embeddings (for example GBERT) compared with the multilingual model, so that the topics are better. | Coherence and a manual topic check for both. |
| 3 | As a reader, I want stance, not only mentions, so that for and against are separated. | Small labelled sample and a classifier with a held-out score. |
| 4 | As a reader, I want the 2021 to 2025 period handled properly, so that the crisis years are covered. | Speeches with party labels for period 20. |
