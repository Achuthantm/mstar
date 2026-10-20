# Grafana dashboards

Grafana loads every `*.json` dashboard in this directory into its `M*` folder
(`../provisioning/dashboards/mstar.yml`) and rereads the directory every 30 seconds.
Panels should use the provisioned data source, uid `prom`:

```json
"datasource": {"type": "prometheus", "uid": "prom"}
```

The metric families are documented in `docs/metrics.rst`.
