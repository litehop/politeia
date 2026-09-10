# politeia

kubeconfig parsing, mutual-TLS (`tokio-rustls`) connector construction, and a
minimal [`hyper`](https://crates.io/crates/hyper) HTTP/1.1 API client with
watch-streaming support for Kubernetes-style API servers.

## Features

- **`parse_kubeconfig`** — read a kubeconfig file and extract TLS credentials
  (`ClientCreds`: server URL, CA cert, client cert, client key) via lightweight
  manual YAML field extraction (no `serde_yaml` dependency).
- **`build_tls_connector`** — build an mTLS `tokio_rustls::TlsConnector` from the
  parsed credentials, installing the ML-KEM-768 hybrid post-quantum key exchange
  provider (`rustls-post-quantum`).
- **`HyperApiClient`** — a minimal HTTP/1.1-over-TLS client that opens a fresh
  connection per request; supports plain requests, custom content types, and
  newline-delimited JSON **watch streams**.
- **`drain_watch_buffer`** — the canonical newline-delimited-JSON watch-event
  parser used by `HyperApiClient::watch_stream`, exported for reuse.

## Usage

```toml
[dependencies]
politeia = { git = "https://github.com/litehop/politeia", tag = "v0.1.0" }
```

```rust
use politeia::{build_tls_connector, parse_kubeconfig, HyperApiClient};

# async fn run() -> anyhow::Result<()> {
let creds = parse_kubeconfig("/path/to/kubeconfig")?;
let connector = build_tls_connector(&creds)?;
let client = HyperApiClient {
    server: creds.server.clone(),
    connector,
    bearer: None,
};

let (status, body) = client.request(hyper::Method::GET, "/api/v1/pods", None).await?;
println!("{status}: {body}");
# Ok(())
# }
```

## License

Licensed under the [Apache License 2.0](LICENSE).
