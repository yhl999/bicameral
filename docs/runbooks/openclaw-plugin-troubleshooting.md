# OpenClaw Plugin Troubleshooting (Bicameral)

This runbook covers common runtime and setup issues specific to the OpenClaw plugin integration (`bicameral`).

## The `id` Config Crash (Port 18789 In Use / Gateway Flapping)

### Symptoms
After adding the Bicameral plugin to `openclaw.json`, the OpenClaw gateway enters a restart loop, fails to start, or shows errors like:
- `Port 18789 is already in use`
- `Gateway failed to start: another gateway instance is already listening on ws://127.0.0.1:18789`
- `[plugins] plugin id mismatch`
- `Invalid config at ~/.openclaw/openclaw.json:\n- plugins.entries.bicameral: Unrecognized key: "id"`

### Root Cause
OpenClaw enforces strict JSON schema validation for plugins (`additionalProperties: false` in `openclaw.plugin.json`).
If you manually include an `"id": "bicameral"` field inside the plugin's configuration block under `plugins.entries` in your main `openclaw.json`, the schema validator will reject the config.
This invalid config can trigger a gateway startup crash loop and leave orphaned Node processes holding the gateway port.

### Fix
Do **not** include the `id` key in the configuration block. The plugin ID is inferred from the object key in `plugins.entries`.

**Incorrect `openclaw.json` (Triggers Crash):**
```json
"plugins": {
  "entries": {
    "bicameral": {
      "id": "bicameral", // ❌ DO NOT DO THIS
      "enabled": true,
      "config": {
        "graphitiBaseUrl": "http://localhost:8000"
      }
    }
  }
}
```

**Correct `openclaw.json`:**
```json
"plugins": {
  "entries": {
    "bicameral": {
      "enabled": true,
      "config": {
        "graphitiBaseUrl": "http://localhost:8000"
      }
    }
  }
}
```

### Recovery Steps (if already crashed)

If the gateway is already stuck in a crashed/flapping state:

1. **Fix the Config**: Remove the `"id"` field from `~/.openclaw/openclaw.json`.
2. **Kill Orphaned Processes**:
   ```bash
   # Find the orphaned processes holding the port
   lsof -i :18789
   # Or using ps
   ps aux | grep node | grep 18789

   # Kill the offending openclaw-gateway PIDs
   kill -9 <PID>
   ```
3. **Restart the Gateway**:
   ```bash
   # If running via launchd (macOS)
   launchctl bootout gui/$UID/ai.openclaw.gateway
   launchctl bootstrap gui/$UID ~/Library/LaunchAgents/ai.openclaw.gateway.plist

   # Or using openclaw CLI
   openclaw gateway restart
   ```

> Legacy deployments may still use the historical plugin key `graphiti-openclaw`. The same strict-schema rule applies: do not include an `id` field inside the plugin entry config block.
