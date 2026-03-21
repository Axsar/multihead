# Mesh Troubleshooting Guide

## Quick Checks

```bash
# Is the daemon running?
multihead mesh status

# Can you see peers on your network?
multihead mesh discover --timeout 10

# What capabilities are registered?
multihead mesh list-peers
```

## Common Issues

### 1. No Peers Discovered

**Symptoms**: `multihead mesh discover` returns empty results.

**Causes & Fixes**:

- **Zeroconf not installed**: Install with `pip install multihead[mesh]`. Without it, mDNS discovery is disabled and only database-backed discovery works.
- **Firewall blocking mDNS**: mDNS uses UDP port 5353. Ensure your firewall allows multicast traffic on this port.
- **Different subnets**: mDNS only works within the same network subnet. For cross-subnet setups, peers discover each other via shared `knowledge.db` presence claims instead. Ensure both nodes point to the same database.
- **Peer not running**: The other node's daemon must be running (`multihead serve`). Check with `curl http://<peer-ip>:7337/health`.

### 2. Daemon Won't Start

**Symptoms**: `multihead serve` fails or exits immediately.

**Check**:
- Port conflict: Default port is 7337. Set `MULTIHEAD_API_PORT` in `.env` to change it.
- Missing data directory: Ensure `MULTIHEAD_DATA_DIR` exists (defaults to `~/.multihead`).
- Run `multihead doctor` for a full diagnostic.

### 3. Authentication Failures (401/403)

**Symptoms**: `list-peers` or task submission returns authentication errors.

**Causes & Fixes**:

- **Mismatched secrets**: All mesh nodes must share the same `MULTIHEAD_MESH_SECRET` in their `.env` files. The secret must be at least 16 characters.
- **No secret set**: If `MULTIHEAD_MESH_SECRET` is not set, authentication is disabled entirely. This is fine for local development but not recommended for shared networks.

### 4. Peers Go Stale / Show "Absent"

**Symptoms**: `mesh status` shows peers as "absent" even though they're running.

**Causes & Fixes**:

- **Heartbeat timeout**: Peers are marked absent after 90 seconds of silence. If a node is under heavy GPU load, heartbeats may be delayed.
- **Network interruption**: Temporary network issues can cause missed heartbeats. Peers re-register automatically when connectivity is restored.
- **PresenceMonitor not started**: In test environments, the presence monitor is intentionally skipped. Ensure you're running a real daemon, not a test server.

### 5. Capabilities Not Showing

**Symptoms**: `list-peers` shows the node but no capabilities.

**Causes & Fixes**:

- **No heads configured**: Check `config/heads.yaml` has valid head entries. Run `multihead heads` to verify.
- **Capabilities auto-register from heads**: The `CapabilityRegistry` populates from head manifests at startup. If heads.yaml is empty or malformed, no capabilities are registered.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIHEAD_API_HOST` | `127.0.0.1` | HTTP API bind address. Use `0.0.0.0` for LAN access. |
| `MULTIHEAD_API_PORT` | `7337` | HTTP API port |
| `MULTIHEAD_MESH_SECRET` | *(none)* | Shared secret for mesh auth (min 16 chars) |
| `MULTIHEAD_DATA_DIR` | `~/.multihead` | Data directory (knowledge.db lives here) |

## Architecture

```
Node A                          Node B
+------------------+           +------------------+
| multihead serve  |           | multihead serve  |
|  - HeadManager   |           |  - HeadManager   |
|  - CapRegistry   |  mDNS    |  - CapRegistry   |
|  - MeshDiscovery |<-------->|  - MeshDiscovery  |
|  - PresenceMon   |  REST    |  - PresenceMon   |
|  - knowledge.db  |<-------->|  - knowledge.db  |
+------------------+           +------------------+
```

**Discovery paths** (tried in order):
1. **mDNS** (Zeroconf): Same-subnet automatic discovery via `_multihead._tcp.local.`
2. **Database presence**: Cross-subnet discovery via shared `knowledge.db` presence claims

## REST Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | No | Liveness probe |
| `GET /v1/health` | No | Mesh health (version, status) |
| `GET /v1/capabilities` | Yes | List node capabilities |
| `POST /v1/tasks` | Yes | Submit task to node |
| `GET /v1/node` | Yes | Node info (id, version) |

## Debugging Commands

```bash
# Full diagnostic
multihead doctor

# Check if a specific peer is reachable
curl http://<peer-ip>:7337/health

# Check peer capabilities (with auth)
curl -H "Authorization: Bearer YOUR_MESH_SECRET" http://<peer-ip>:7337/v1/capabilities

# View presence claims in knowledge DB
multihead knowledge claims --status accepted | grep mesh.presence
```
