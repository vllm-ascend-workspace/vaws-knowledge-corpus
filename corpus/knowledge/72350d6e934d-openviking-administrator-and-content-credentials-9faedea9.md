# OpenViking administrator and content credentials [9faedea9]

# OpenViking administrator and content credentials

For the local OpenViking 0.4.19 API-key configuration used by vaws-knowledge 0.3.1, provision a tenant data user with the root administrator key, then use that tenant key for document writes, dense OVPack export and import. The root key belongs to account administration.

The package's ARM64 macOS CPU build was exercised with OpenViking SDK 0.1.10 and FastEmbed 0.8.0. A Markdown corpus at an exact Git commit produced a validated dense OVPack. Local credential files use restrictive permissions and status responses omit the keys.

This observation covers the local build and tenant API path. It does not establish Windows behavior, large-corpus performance or Ascend model correctness.
