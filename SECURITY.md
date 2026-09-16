# Security policy

## Supported versions

OPHANIM is pre-production research software. Security fixes are applied to the
latest tagged release only.

## Reporting a vulnerability

Do not open a public issue containing exploit details, credentials, private TEC
data, or local paths. Use the repository owner's private GitHub contact or a
private security advisory. Include the affected version, reproduction steps,
and the smallest safe example possible.

The desktop server is designed for loopback use. Do not publish its port to an
untrusted network, and never commit `.env`, `var/`, SQLite databases, Zarr
stores, downloaded source artifacts, or model artifacts.
