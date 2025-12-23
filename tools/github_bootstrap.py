#!/usr/bin/env python3
"""Bootstrap and configure a GitHub repo via the REST API.

This script creates a repository (user or org) and applies as many settings
as the API allows: repo metadata, topics, labels, Actions permissions,
security settings, environments, and branch protections.

It is designed to be idempotent. Re-run safely to converge settings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Iterable

API_BASE = "https://api.github.com"
DEFAULT_API_VERSION = "2022-11-28"

DEFAULT_LABELS = [
    {"name": "bug", "color": "d73a4a", "description": "Something is not working"},
    {"name": "enhancement", "color": "a2eeef", "description": "New feature or request"},
    {"name": "documentation", "color": "0075ca", "description": "Docs or examples"},
    {"name": "security", "color": "b60205", "description": "Security-related issue"},
    {"name": "tests", "color": "0e8a16", "description": "Test coverage or CI"},
    {"name": "long-run", "color": "5319e7", "description": "Long-horizon reliability"},
]

DEFAULT_CONFIG = {
    "create": {
        "auto_init": False,
        "gitignore_template": "",
        "license_template": "",
        "team_id": None,
    },
    "repo": {
        "name": "",
        "owner": "",
        "visibility": "public",
        "description": "Long-horizon LLM training and testing harness",
        "homepage": "",
        "topics": [
            "llm",
            "long-horizon",
            "agent",
            "testing",
            "coverage",
            "reflex",
            "neo4j",
            "playwright",
            "typst",
            "automation",
        ],
        "settings": {
            "has_issues": True,
            "has_projects": True,
            "has_wiki": False,
            "has_discussions": True,
            "is_template": False,
            "allow_squash_merge": True,
            "allow_merge_commit": False,
            "allow_rebase_merge": True,
            "allow_auto_merge": True,
            "delete_branch_on_merge": True,
            "allow_update_branch": True,
            "use_squash_pr_title_as_default": True,
            "web_commit_signoff_required": True,
            "squash_merge_commit_title": "PR_TITLE",
            "squash_merge_commit_message": "PR_BODY",
            "merge_commit_title": "PR_TITLE",
            "merge_commit_message": "PR_BODY",
        },
        "security_and_analysis": {
            "advanced_security": "enabled",
            "secret_scanning": "enabled",
            "secret_scanning_push_protection": "enabled",
            "dependabot_security_updates": "enabled",
        },
    },
    "actions": {
        "enabled": True,
        "allowed_actions": "all",
        "default_workflow_permissions": "read",
        "can_approve_pull_request_reviews": False,
    },
    "actions_access": None,
    "actions_artifact_and_log_retention": None,
    "actions_cache_retention_limit": None,
    "actions_cache_storage_limit": None,
    "actions_fork_pr_contributor_approval": None,
    "actions_fork_pr_workflows_private_repos": None,
    "actions_oidc_subject": None,
    "actions_selected": {
        "github_owned_allowed": True,
        "verified_allowed": True,
        "patterns_allowed": [],
    },
    "actions_variables": [],
    "actions_secrets": [],
    "check_suite_preferences": None,
    "code_scanning_default_setup": None,
    "private_vulnerability_reporting": None,
    "labels": DEFAULT_LABELS,
    "collaborators": [],
    "teams": [],
    "webhooks": [],
    "autolinks": [],
    "deploy_keys": [],
    "tag_protection": [],
    "custom_properties": [],
    "rulesets": [],
    "branch_protection": {
        "branch": "main",
        "required_status_checks": {
            "strict": True,
            "contexts": ["maxcov / maxcov"],
        },
        "enforce_admins": True,
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": True,
        },
        "restrictions": None,
        "required_linear_history": True,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "block_creations": False,
        "required_conversation_resolution": True,
        "lock_branch": False,
        "allow_fork_syncing": True,
    },
    "environments": [
        {
            "name": "production",
            "wait_timer": 0,
            "reviewers": [],
            "deployment_branch_policy": {
                "protected_branches": True,
                "custom_branch_policies": False,
            },
        }
    ],
    "pages": None,
    "enable_dependabot_alerts": True,
    "enable_automated_security_fixes": True,
}


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _api_request(
    token: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    accept: str | None = None,
    expected: Iterable[int] = (200, 201, 202, 204),
) -> dict[str, Any] | None:
    url = f"{API_BASE}{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": accept or "application/vnd.github+json",
        "X-GitHub-Api-Version": DEFAULT_API_VERSION,
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            raw = response.read().decode("utf-8")
            if not raw:
                return None
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else ""
        detail = raw
        try:
            if raw:
                detail_json = json.loads(raw)
                detail = json.dumps(detail_json, indent=2)
        except Exception:
            pass
        if exc.code not in expected:
            raise ApiError(f"{method} {path} failed: {exc.code} {detail}", exc.code)
        return None


def _get_login(token: str) -> str:
    data = _api_request(token, "GET", "/user")
    if not data or "login" not in data:
        raise ApiError("Unable to determine GitHub login")
    return data["login"]


def _create_repo(token: str, owner: str, is_org: bool, repo_payload: dict[str, Any]) -> None:
    path = f"/orgs/{owner}/repos" if is_org else "/user/repos"
    _api_request(token, "POST", path, repo_payload, expected=(201, 422))


def _update_repo(token: str, owner: str, repo: str, payload: dict[str, Any]) -> None:
    _api_request(token, "PATCH", f"/repos/{owner}/{repo}", payload)


def _delete_repo(token: str, owner: str, repo: str) -> None:
    _api_request(token, "DELETE", f"/repos/{owner}/{repo}", expected=(204, 404))


def _set_topics(token: str, owner: str, repo: str, topics: list[str]) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/topics",
        {"names": topics},
    )


def _set_actions_permissions(token: str, owner: str, repo: str, payload: dict[str, Any]) -> None:
    _api_request(token, "PUT", f"/repos/{owner}/{repo}/actions/permissions", payload)


def _set_actions_workflow_permissions(
    token: str, owner: str, repo: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/actions/permissions/workflow",
        payload,
    )


def _set_actions_selected(
    token: str, owner: str, repo: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/actions/permissions/selected-actions",
        payload,
    )


def _set_actions_access(token: str, owner: str, repo: str, payload: dict[str, Any]) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/actions/permissions/access",
        payload,
    )


def _set_actions_artifact_retention(
    token: str, owner: str, repo: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/actions/permissions/artifact-and-log-retention",
        payload,
    )


def _set_actions_cache_retention_limit(
    token: str, owner: str, repo: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/actions/cache/retention-limit",
        payload,
    )


def _set_actions_cache_storage_limit(
    token: str, owner: str, repo: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/actions/cache/storage-limit",
        payload,
    )


def _set_actions_fork_pr_contributor_approval(
    token: str, owner: str, repo: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/actions/permissions/fork-pr-contributor-approval",
        payload,
    )


def _set_actions_fork_pr_workflows_private_repos(
    token: str, owner: str, repo: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/actions/permissions/fork-pr-workflows-private-repos",
        payload,
    )


def _set_actions_oidc_subject(
    token: str, owner: str, repo: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/actions/oidc/customization/sub",
        payload,
    )


def _set_check_suite_preferences(
    token: str, owner: str, repo: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PATCH",
        f"/repos/{owner}/{repo}/check-suites/preferences",
        payload,
    )


def _set_code_scanning_default_setup(
    token: str, owner: str, repo: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PATCH",
        f"/repos/{owner}/{repo}/code-scanning/default-setup",
        payload,
    )


def _set_private_vulnerability_reporting(
    token: str, owner: str, repo: str, enabled: bool
) -> None:
    method = "PUT" if enabled else "DELETE"
    _api_request(token, method, f"/repos/{owner}/{repo}/private-vulnerability-reporting")


def _set_custom_properties(
    token: str, owner: str, repo: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PATCH",
        f"/repos/{owner}/{repo}/properties/values",
        payload,
    )


def _set_security_toggle(token: str, owner: str, repo: str, endpoint: str) -> None:
    _api_request(token, "PUT", f"/repos/{owner}/{repo}/{endpoint}", expected=(204,))


def _ensure_label(token: str, owner: str, repo: str, label: dict[str, Any]) -> None:
    try:
        _api_request(token, "POST", f"/repos/{owner}/{repo}/labels", label)
    except ApiError as exc:
        if exc.status != 422:
            raise
        name = label.get("name")
        if not name:
            return
        _api_request(token, "PATCH", f"/repos/{owner}/{repo}/labels/{name}", label)


def _set_branch_protection(
    token: str, owner: str, repo: str, branch: str, payload: dict[str, Any]
) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/branches/{branch}/protection",
        payload,
    )


def _set_branch_required_signatures(
    token: str, owner: str, repo: str, branch: str, required: bool
) -> None:
    method = "POST" if required else "DELETE"
    _api_request(
        token,
        method,
        f"/repos/{owner}/{repo}/branches/{branch}/protection/required_signatures",
        expected=(200, 204, 404),
    )


def _set_environment(token: str, owner: str, repo: str, env: dict[str, Any]) -> None:
    name = env.get("name")
    if not name:
        return
    payload = {k: v for k, v in env.items() if k != "name"}
    _api_request(token, "PUT", f"/repos/{owner}/{repo}/environments/{name}", payload)


def _create_pages(token: str, owner: str, repo: str, payload: dict[str, Any]) -> None:
    _api_request(
        token,
        "POST",
        f"/repos/{owner}/{repo}/pages",
        payload,
        expected=(201, 202, 204, 409, 422),
    )


def _update_pages(token: str, owner: str, repo: str, payload: dict[str, Any]) -> None:
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/pages",
        payload,
    )


def _list_webhooks(token: str, owner: str, repo: str) -> list[dict[str, Any]]:
    data = _api_request(token, "GET", f"/repos/{owner}/{repo}/hooks") or []
    if not isinstance(data, list):
        return []
    return data


def _ensure_webhook(token: str, owner: str, repo: str, hook: dict[str, Any]) -> None:
    hook_id = hook.get("id")
    if hook_id:
        _api_request(
            token,
            "PATCH",
            f"/repos/{owner}/{repo}/hooks/{hook_id}",
            hook,
        )
        return
    try:
        _api_request(token, "POST", f"/repos/{owner}/{repo}/hooks", hook)
    except ApiError as exc:
        if exc.status != 422:
            raise
        target_url = (hook.get("config") or {}).get("url")
        if not target_url:
            return
        for existing in _list_webhooks(token, owner, repo):
            if (existing.get("config") or {}).get("url") == target_url:
                _api_request(
                    token,
                    "PATCH",
                    f"/repos/{owner}/{repo}/hooks/{existing.get('id')}",
                    hook,
                )
                return


def _list_autolinks(token: str, owner: str, repo: str) -> list[dict[str, Any]]:
    data = _api_request(token, "GET", f"/repos/{owner}/{repo}/autolinks") or []
    if not isinstance(data, list):
        return []
    return data


def _ensure_autolink(token: str, owner: str, repo: str, autolink: dict[str, Any]) -> None:
    key_prefix = autolink.get("key_prefix")
    url_template = autolink.get("url_template")
    for existing in _list_autolinks(token, owner, repo):
        if existing.get("key_prefix") == key_prefix and existing.get("url_template") == url_template:
            return
    _api_request(token, "POST", f"/repos/{owner}/{repo}/autolinks", autolink)


def _list_deploy_keys(token: str, owner: str, repo: str) -> list[dict[str, Any]]:
    data = _api_request(token, "GET", f"/repos/{owner}/{repo}/keys") or []
    if not isinstance(data, list):
        return []
    return data


def _ensure_deploy_key(token: str, owner: str, repo: str, deploy_key: dict[str, Any]) -> None:
    title = deploy_key.get("title")
    key = deploy_key.get("key")
    for existing in _list_deploy_keys(token, owner, repo):
        if title and existing.get("title") == title:
            return
        if key and existing.get("key") == key:
            return
    _api_request(token, "POST", f"/repos/{owner}/{repo}/keys", deploy_key)


def _list_tag_protection(token: str, owner: str, repo: str) -> list[dict[str, Any]]:
    try:
        data = _api_request(token, "GET", f"/repos/{owner}/{repo}/tags/protection") or []
    except ApiError as exc:
        if exc.status in (404, 410):
            return []
        raise
    if not isinstance(data, list):
        return []
    return data


def _ensure_tag_protection(token: str, owner: str, repo: str, tag: dict[str, Any]) -> None:
    pattern = tag.get("pattern")
    if not pattern:
        return
    for existing in _list_tag_protection(token, owner, repo):
        if existing.get("pattern") == pattern:
            return
    _api_request(token, "POST", f"/repos/{owner}/{repo}/tags/protection", tag)


def _add_collaborator(
    token: str, owner: str, repo: str, collaborator: dict[str, Any]
) -> None:
    username = collaborator.get("username")
    if not username:
        return
    payload = {}
    permission = collaborator.get("permission")
    if permission:
        payload["permission"] = permission
    _api_request(
        token,
        "PUT",
        f"/repos/{owner}/{repo}/collaborators/{username}",
        payload if payload else None,
        expected=(201, 204),
    )


def _add_team(token: str, owner: str, repo: str, team: dict[str, Any]) -> None:
    slug = team.get("slug")
    if not slug:
        return
    payload = {}
    permission = team.get("permission")
    if permission:
        payload["permission"] = permission
    _api_request(
        token,
        "PUT",
        f"/orgs/{owner}/teams/{slug}/repos/{owner}/{repo}",
        payload if payload else None,
        expected=(204,),
    )


def _list_rulesets(token: str, owner: str, repo: str) -> list[dict[str, Any]]:
    data = _api_request(token, "GET", f"/repos/{owner}/{repo}/rulesets") or []
    if not isinstance(data, list):
        return []
    return data


def _ensure_ruleset(token: str, owner: str, repo: str, ruleset: dict[str, Any]) -> None:
    ruleset_id = ruleset.get("id")
    if ruleset_id:
        _api_request(
            token,
            "PUT",
            f"/repos/{owner}/{repo}/rulesets/{ruleset_id}",
            ruleset,
        )
        return
    name = ruleset.get("name")
    for existing in _list_rulesets(token, owner, repo):
        if name and existing.get("name") == name:
            _api_request(
                token,
                "PUT",
                f"/repos/{owner}/{repo}/rulesets/{existing.get('id')}",
                ruleset,
            )
            return
    _api_request(token, "POST", f"/repos/{owner}/{repo}/rulesets", ruleset)


def _set_actions_variable(token: str, owner: str, repo: str, variable: dict[str, Any]) -> None:
    name = variable.get("name")
    if not name:
        return
    try:
        _api_request(
            token,
            "POST",
            f"/repos/{owner}/{repo}/actions/variables",
            {"name": name, "value": variable.get("value", "")},
        )
    except ApiError as exc:
        if exc.status != 409:
            raise
        _api_request(
            token,
            "PATCH",
            f"/repos/{owner}/{repo}/actions/variables/{name}",
            {"name": name, "value": variable.get("value", "")},
        )


def _encrypt_secret(public_key: str, secret_value: str) -> str:
    try:
        from nacl import encoding, public  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ApiError("PyNaCl is required to set secrets") from exc

    key = public.PublicKey(public_key.encode("utf-8"), encoding.Base64Encoder())
    sealed_box = public.SealedBox(key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return encoding.Base64Encoder().encode(encrypted).decode("utf-8")


def _set_actions_secret(token: str, owner: str, repo: str, secret: dict[str, Any]) -> None:
    name = secret.get("name")
    value = secret.get("value")
    if not name or value is None:
        return
    key = _api_request(token, "GET", f"/repos/{owner}/{repo}/actions/secrets/public-key")
    if not key:
        raise ApiError("Unable to fetch public key for secrets")
    encrypted_value = _encrypt_secret(key["key"], value)
    payload = {"encrypted_value": encrypted_value, "key_id": key["key_id"]}
    _api_request(token, "PUT", f"/repos/{owner}/{repo}/actions/secrets/{name}", payload)


def _repo_payload(repo_config: dict[str, Any], create_config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "name": repo_config.get("name"),
        "description": repo_config.get("description"),
        "homepage": repo_config.get("homepage") or None,
        "private": repo_config.get("visibility") == "private",
        "visibility": repo_config.get("visibility"),
        "auto_init": False,
        "has_issues": repo_config.get("settings", {}).get("has_issues"),
        "has_projects": repo_config.get("settings", {}).get("has_projects"),
        "has_wiki": repo_config.get("settings", {}).get("has_wiki"),
        "has_discussions": repo_config.get("settings", {}).get("has_discussions"),
        "is_template": repo_config.get("settings", {}).get("is_template"),
    }
    create_fields = {
        "auto_init": create_config.get("auto_init"),
        "gitignore_template": create_config.get("gitignore_template") or None,
        "license_template": create_config.get("license_template") or None,
        "team_id": create_config.get("team_id"),
    }
    payload.update({k: v for k, v in create_fields.items() if v not in (None, "")})
    return {k: v for k, v in payload.items() if v is not None}


def _repo_update_payload(repo_config: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    payload.update(repo_config.get("settings", {}))
    allow_merge_commit = payload.get("allow_merge_commit")
    if allow_merge_commit is False:
        payload.pop("merge_commit_title", None)
        payload.pop("merge_commit_message", None)
    allow_squash_merge = payload.get("allow_squash_merge")
    if allow_squash_merge is False:
        payload.pop("squash_merge_commit_title", None)
        payload.pop("squash_merge_commit_message", None)
    for key in (
        "description",
        "homepage",
        "visibility",
        "default_branch",
        "allow_forking",
        "archived",
    ):
        if key in repo_config and repo_config.get(key) not in (None, ""):
            payload[key] = repo_config[key]
    security = repo_config.get("security_and_analysis") or {}
    if repo_config.get("visibility") == "public":
        security = dict(security)
        security.pop("advanced_security", None)
    if security:
        sec_payload: dict[str, Any] = {}
        if "advanced_security" in security:
            sec_payload["advanced_security"] = {"status": security.get("advanced_security")}
        if "secret_scanning" in security:
            sec_payload["secret_scanning"] = {"status": security.get("secret_scanning")}
        if "secret_scanning_push_protection" in security:
            sec_payload["secret_scanning_push_protection"] = {
                "status": security.get("secret_scanning_push_protection")
            }
        if "dependabot_security_updates" in security:
            sec_payload["dependabot_security_updates"] = {
                "status": security.get("dependabot_security_updates")
            }
        if sec_payload:
            payload["security_and_analysis"] = sec_payload
    return payload


def _log(message: str) -> None:
    print(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and configure a GitHub repo")
    parser.add_argument("--config", default="tools/github_bootstrap_config.json")
    parser.add_argument("--owner", default="")
    parser.add_argument("--org", default="", help="Create repo under this org")
    parser.add_argument("--repo", default="")
    parser.add_argument("--auto-init", action="store_true", help="Initialize repo on create")
    parser.add_argument("--delete", action="store_true", help="Delete the target repo")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive operations")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-create", action="store_true")
    parser.add_argument("--skip-branch-protection", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--skip-actions", action="store_true")
    parser.add_argument("--skip-security", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_PAT") or os.environ.get("GITHUB_TOKEN")
    if not token:
        _log("GITHUB_PAT or GITHUB_TOKEN is required")
        return 1

    config = DEFAULT_CONFIG
    if args.config and os.path.exists(args.config):
        config = _merge_dicts(config, _load_json(args.config))

    create_config = config.get("create") or {}
    repo_config = config.get("repo", {})
    if args.repo:
        repo_config["name"] = args.repo
    if args.owner:
        repo_config["owner"] = args.owner
    if args.auto_init:
        create_config["auto_init"] = True

    owner = repo_config.get("owner") or args.org
    if not owner:
        owner = _get_login(token)
        repo_config["owner"] = owner

    repo_name = repo_config.get("name")
    if not repo_name:
        _log("Repository name is required")
        return 1

    is_org = bool(args.org)
    _log(f"Target repo: {owner}/{repo_name}")

    if args.dry_run:
        _log("Dry-run enabled. No changes will be made.")
        return 0

    if args.delete:
        if not args.yes:
            _log("Refusing to delete without --yes confirmation.")
            return 1
        _log("Deleting repository")
        _delete_repo(token, owner, repo_name)
        _log("Delete complete")
        return 0

    errors: list[str] = []

    def run_step(label: str, func) -> None:
        try:
            _log(label)
            func()
        except ApiError as exc:
            errors.append(f"{label}: {exc}")
            _log(f"WARNING: {label} failed: {exc}")

    if not args.skip_create:
        run_step(
            "Creating repository",
            lambda: _create_repo(
                token, owner, is_org, _repo_payload(repo_config, create_config)
            ),
        )

    run_step(
        "Updating repository settings",
        lambda: _update_repo(token, owner, repo_name, _repo_update_payload(repo_config)),
    )

    topics = repo_config.get("topics") or []
    if topics:
        run_step("Setting topics", lambda: _set_topics(token, owner, repo_name, topics))

    if not args.skip_actions:
        actions = config.get("actions") or {}
        if actions:
            run_step(
                "Configuring Actions permissions",
                lambda: _set_actions_permissions(token, owner, repo_name, actions),
            )
        workflow = {
            "default_workflow_permissions": actions.get("default_workflow_permissions", "read"),
            "can_approve_pull_request_reviews": actions.get(
                "can_approve_pull_request_reviews", False
            ),
        }
        run_step(
            "Configuring Actions workflow permissions",
            lambda: _set_actions_workflow_permissions(token, owner, repo_name, workflow),
        )
        if actions.get("allowed_actions") == "selected":
            selected = config.get("actions_selected") or {}
            run_step(
                "Configuring selected Actions allowlist",
                lambda: _set_actions_selected(token, owner, repo_name, selected),
            )
        actions_access = config.get("actions_access")
        if actions_access and repo_config.get("visibility") != "public":
            run_step(
                "Configuring Actions workflow access",
                lambda: _set_actions_access(token, owner, repo_name, actions_access),
            )
        actions_retention = config.get("actions_artifact_and_log_retention")
        if actions_retention:
            run_step(
                "Configuring Actions artifact/log retention",
                lambda: _set_actions_artifact_retention(
                    token, owner, repo_name, actions_retention
                ),
            )
        actions_cache_retention = config.get("actions_cache_retention_limit")
        if actions_cache_retention:
            run_step(
                "Configuring Actions cache retention limit",
                lambda: _set_actions_cache_retention_limit(
                    token, owner, repo_name, actions_cache_retention
                ),
            )
        actions_cache_storage = config.get("actions_cache_storage_limit")
        if actions_cache_storage:
            run_step(
                "Configuring Actions cache storage limit",
                lambda: _set_actions_cache_storage_limit(
                    token, owner, repo_name, actions_cache_storage
                ),
            )
        fork_pr_approval = config.get("actions_fork_pr_contributor_approval")
        if fork_pr_approval:
            run_step(
                "Configuring Actions fork PR approval policy",
                lambda: _set_actions_fork_pr_contributor_approval(
                    token, owner, repo_name, fork_pr_approval
                ),
            )
        fork_pr_private = config.get("actions_fork_pr_workflows_private_repos")
        if fork_pr_private and repo_config.get("visibility") != "public":
            run_step(
                "Configuring Actions fork PR settings (private repos)",
                lambda: _set_actions_fork_pr_workflows_private_repos(
                    token, owner, repo_name, fork_pr_private
                ),
            )
        oidc_subject = config.get("actions_oidc_subject")
        if oidc_subject:
            run_step(
                "Configuring Actions OIDC subject claim",
                lambda: _set_actions_oidc_subject(token, owner, repo_name, oidc_subject),
            )
        for variable in config.get("actions_variables") or []:
            name = variable.get("name") or "unknown"
            run_step(
                f"Setting Actions variable {name}",
                lambda var=variable: _set_actions_variable(token, owner, repo_name, var),
            )
        for secret in config.get("actions_secrets") or []:
            name = secret.get("name") or "unknown"
            run_step(
                f"Setting Actions secret {name}",
                lambda sec=secret: _set_actions_secret(token, owner, repo_name, sec),
            )

    if not args.skip_security:
        if config.get("enable_dependabot_alerts"):
            run_step(
                "Enabling Dependabot alerts",
                lambda: _set_security_toggle(token, owner, repo_name, "vulnerability-alerts"),
            )
        if config.get("enable_automated_security_fixes"):
            run_step(
                "Enabling automated security fixes",
                lambda: _set_security_toggle(token, owner, repo_name, "automated-security-fixes"),
            )
        if config.get("private_vulnerability_reporting") is not None:
            enabled = bool(config.get("private_vulnerability_reporting"))
            run_step(
                "Configuring private vulnerability reporting",
                lambda: _set_private_vulnerability_reporting(token, owner, repo_name, enabled),
            )
        code_scanning = config.get("code_scanning_default_setup")
        if code_scanning:
            run_step(
                "Configuring code scanning default setup",
                lambda: _set_code_scanning_default_setup(token, owner, repo_name, code_scanning),
            )

    check_suite_prefs = config.get("check_suite_preferences")
    if check_suite_prefs:
        run_step(
            "Configuring check suite preferences",
            lambda: _set_check_suite_preferences(token, owner, repo_name, check_suite_prefs),
        )

    if not args.skip_labels:
        for label in config.get("labels") or []:
            name = label.get("name") or "unknown"
            run_step(
                f"Ensuring label {name}",
                lambda lab=label: _ensure_label(token, owner, repo_name, lab),
            )

    for collaborator in config.get("collaborators") or []:
        name = collaborator.get("username") or "unknown"
        run_step(
            f"Adding collaborator {name}",
            lambda collab=collaborator: _add_collaborator(token, owner, repo_name, collab),
        )

    for team in config.get("teams") or []:
        name = team.get("slug") or "unknown"
        run_step(
            f"Adding team {name}",
            lambda team_cfg=team: _add_team(token, owner, repo_name, team_cfg),
        )

    for hook in config.get("webhooks") or []:
        hook_name = (hook.get("name") or "web").strip()
        run_step(
            f"Ensuring webhook {hook_name}",
            lambda hook_cfg=hook: _ensure_webhook(token, owner, repo_name, hook_cfg),
        )

    for autolink in config.get("autolinks") or []:
        prefix = autolink.get("key_prefix") or "autolink"
        run_step(
            f"Ensuring autolink {prefix}",
            lambda link_cfg=autolink: _ensure_autolink(token, owner, repo_name, link_cfg),
        )

    for deploy_key in config.get("deploy_keys") or []:
        title = deploy_key.get("title") or "deploy-key"
        run_step(
            f"Ensuring deploy key {title}",
            lambda key_cfg=deploy_key: _ensure_deploy_key(token, owner, repo_name, key_cfg),
        )

    pages = config.get("pages")
    if isinstance(pages, dict) and ("create" in pages or "update" in pages):
        if pages.get("create"):
            run_step(
                "Creating GitHub Pages",
                lambda: _create_pages(token, owner, repo_name, pages["create"]),
            )
        if pages.get("update"):
            run_step(
                "Updating GitHub Pages",
                lambda: _update_pages(token, owner, repo_name, pages["update"]),
            )
    elif pages:
        run_step(
            "Configuring GitHub Pages",
            lambda: _create_pages(token, owner, repo_name, pages),
        )

    for env in config.get("environments") or []:
        name = env.get("name") or "unknown"
        run_step(
            f"Configuring environment {name}",
            lambda env_cfg=env: _set_environment(token, owner, repo_name, env_cfg),
        )

    for ruleset in config.get("rulesets") or []:
        name = ruleset.get("name") or "ruleset"
        run_step(
            f"Ensuring ruleset {name}",
            lambda ruleset_cfg=ruleset: _ensure_ruleset(token, owner, repo_name, ruleset_cfg),
        )

    custom_properties = config.get("custom_properties") or []
    if custom_properties:
        run_step(
            "Setting custom properties",
            lambda: _set_custom_properties(
                token, owner, repo_name, {"properties": custom_properties}
            ),
        )

    for tag in config.get("tag_protection") or []:
        pattern = tag.get("pattern") or "tag"
        run_step(
            f"Ensuring tag protection {pattern}",
            lambda tag_cfg=tag: _ensure_tag_protection(token, owner, repo_name, tag_cfg),
        )

    if not args.skip_branch_protection:
        bp = config.get("branch_protection") or {}
        branch = bp.get("branch") or "main"
        if bp:
            required_signatures = bp.get("required_signatures")
            payload = {
                k: v for k, v in bp.items() if k not in ("branch", "required_signatures")
            }
            run_step(
                "Applying branch protection",
                lambda: _set_branch_protection(token, owner, repo_name, branch, payload),
            )
            if required_signatures is not None:
                run_step(
                    "Configuring required signatures",
                    lambda: _set_branch_required_signatures(
                        token, owner, repo_name, branch, bool(required_signatures)
                    ),
                )

    if errors:
        _log("Bootstrap completed with warnings:")
        for error in errors:
            _log(f"- {error}")
        return 1

    _log("Bootstrap complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
