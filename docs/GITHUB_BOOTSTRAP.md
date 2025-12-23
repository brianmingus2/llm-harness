# GitHub Bootstrap

Use `tools/github_bootstrap.py` to create and fully configure a repository via the GitHub API. It sets repo metadata, topics, labels, Actions permissions, security features, environments, Pages (optional), rulesets, collaborators, webhooks, deploy keys, tag protection, custom properties, and branch protections.

## Prerequisites

- `GITHUB_PAT` or `GITHUB_TOKEN` with repo/admin permissions
- Optional: `pynacl` if you want to set encrypted Actions secrets

## Quick start

```bash
export GITHUB_PAT=YOUR_TOKEN

python tools/github_bootstrap.py \
  --config tools/github_bootstrap_config.json \
  --org YOUR_ORG \
  --repo llm-harness
```

If you are using a personal account, omit `--org` and use `--owner` or let the script infer the owner.

To auto-initialize a repo (so branch protections can be applied immediately):

```bash
python tools/github_bootstrap.py --config tools/github_bootstrap_config.json --auto-init --repo your-repo
```

## Configuration

Edit `tools/github_bootstrap_config.json` and replace `REPLACE_ME`. The config covers:

- Repo metadata and visibility
- Create-time options (auto-init, gitignore template, license template)
- Merge strategy policies
- Topics and labels
- Actions permissions, workflow access, OIDC subject, cache limits, and retention
- Check suite preferences
- Code scanning default setup and private vulnerability reporting
- Security alerts and scanning
- Environment protection rules
- Collaborators, teams, webhooks, autolinks, deploy keys, tag protection
- Custom properties (org-managed repositories)
- Rulesets (optional)
- Branch protection (required checks, reviews, required signatures, etc.)

## Teardown

Delete a repo (requires explicit confirmation):

```bash
python tools/github_bootstrap.py --config tools/github_bootstrap_config.json --repo your-repo --delete --yes
```

## Notes

- Branch protection requires the target branch to exist.
- Actions secrets require `pynacl` for encryption.
- Some security features require GitHub Advanced Security or may be plan-limited.
