# Security Guidelines

## Secrets Management

This project strictly follows a "No Secrets in Git" policy.

- **NEVER** commit `.env` files or files containing actual API keys, passwords, or tokens.
- Use `.env.example` files to document required environment variables with **placeholder** values.
- If you accidentally commit a secret:
  1. Revoke the secret immediately.
  2. Rotate the key/token.
  3. Remove the secret from the commit history (using `git filter-branch` or BFG Repo-Cleaner if deeply buried, or `git rm --cached` if recent).

## API Keys

- **Supabase**: Use `SUPABASE_ANON_KEY` for client-side (public) operations. Do not expose `SUPABASE_SERVICE_ROLE_KEY` in the frontend.
- **InfluxDB**: manage tokens via the InfluxDB UI or CLI.
- **Railway**: Manage secrets via the Railway Dashboard variables.

## Reporting Vulnerabilities

If you find a security vulnerability, please do not open a public issue. Contact the repository maintainers directly.
