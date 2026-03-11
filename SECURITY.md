# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
|---------|--------------------|
| 0.x     | :white_check_mark: |


## Reporting a Vulnerability

Please use GitHub's Private Vulnerability Reporting mechanism found in the Security section of this repo.

## Deployment Threat Model Assumptions

### Single-User, Private Instance

This codebase assumes a single-user, private deployment (Archibald instance for Yuan).
The following threats are **documented but not defended against** because they assume
external attacker control of the deployment environment:

#### SSRF via Environment Variables
- If an attacker can set `GRAPHITI_MCP_URL` or `GRAPHITI_BASE_URL` env vars, they can
  redirect MCP requests to arbitrary services.
- **Assumption:** environment variables are controlled by the operator and not exposed
  to untrusted processes.
- **Mitigation:** (none shipped) validate URLs against a whitelist if this assumption
  changes.

#### Session ID Fixation
- If an attacker can manipulate MCP session IDs in responses, they could cause the router
  to reuse stale sessions or sessions from other principals.
- **Assumption:** the MCP server is trusted and not compromised.
- **Mitigation:** (none shipped) implement session pinning / ephemeral session creation
  if deployment expands to multi-tenant or untrusted MCP servers.

### If Open-Sourcing
If this codebase becomes a multi-tenant or open-source service, revisit:
1. Environment variable source validation
2. Session ID pinning / freshness checks
3. URL whitelist enforcement for all MCP service URLs
