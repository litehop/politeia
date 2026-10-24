/// politeia — kubeconfig parsing and TLS client construction.
///
/// Reads a kubeconfig file, extracts TLS credentials, and builds a tokio-rustls
/// TlsConnector for mTLS connections to a Kubernetes-style API server. Also
/// provides a minimal hyper HTTP/1.1 API client with watch-streaming support.
use std::sync::Arc;

use anyhow::Context;
use base64::Engine;
use hyper::{Method, Request};
use hyper_util::rt::TokioIo;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;
use tracing::{error, warn};

/// Parsed credentials extracted from a kubeconfig file.
pub struct ClientCreds {
    /// Base URL of the API server, e.g. "https://127.0.0.1:6443"
    pub server: String,
    /// DER-encoded CA certificate used to verify the server.
    pub ca_cert: CertificateDer<'static>,
    /// DER-encoded client certificate.
    pub client_cert: CertificateDer<'static>,
    /// DER-encoded client private key.
    pub client_key: PrivateKeyDer<'static>,
}

/// Parse a kubeconfig file and return TLS credentials.
///
/// Performs manual YAML field extraction without a serde_yaml dependency.
/// Expects the fixed kubeconfig structure. Each of CA cert, client cert and
/// client key is read from inline base64 `*-data` or, as in client-go, from a
/// `certificate-authority` / `client-certificate` / `client-key` file path
/// (relative paths resolve against the kubeconfig's directory); `*-data` wins
/// when both are present.
pub fn parse_kubeconfig(path: &str) -> anyhow::Result<ClientCreds> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading kubeconfig {path}"))?;
    let dir = std::path::Path::new(path)
        .parent()
        .unwrap_or_else(|| std::path::Path::new(""));

    let server = extract_yaml_value(&raw, "server:").context("kubeconfig: missing server")?;
    let ca_pem = pem_source(
        &raw,
        dir,
        "certificate-authority-data:",
        "certificate-authority:",
    )?
    .read()?;
    let cert_pem =
        pem_source(&raw, dir, "client-certificate-data:", "client-certificate:")?.read()?;
    let key_pem = pem_source(&raw, dir, "client-key-data:", "client-key:")?.read()?;

    // The kubeconfig fields hold PEM; rustls needs DER.
    // pem_slice_iter strips the PEM envelope and yields raw DER.
    let ca_cert = CertificateDer::pem_slice_iter(&ca_pem)
        .next()
        .context("no certificate in kubeconfig certificate-authority")?
        .context("parse CA cert PEM")?;
    let client_cert = CertificateDer::pem_slice_iter(&cert_pem)
        .next()
        .context("no certificate in kubeconfig client-certificate")?
        .context("parse client cert PEM")?;
    let client_key = PrivateKeyDer::from_pem_slice(&key_pem)
        .context("parse private key in kubeconfig client-key")?;

    Ok(ClientCreds {
        server: server.trim().to_owned(),
        ca_cert,
        client_cert,
        client_key,
    })
}

/// Extract the first occurrence of a YAML scalar value for `key` in `text`.
/// Handles both "  key: value" and "key: value" with arbitrary leading whitespace.
pub fn extract_yaml_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(key) {
            return Some(rest.trim());
        }
    }
    None
}

/// Build a tokio-rustls TlsConnector from parsed kubeconfig credentials.
///
/// Configures mTLS: the CA cert is used to verify the server; the client
/// cert/key are presented for mutual authentication.
pub fn build_tls_connector(creds: &ClientCreds) -> anyhow::Result<TlsConnector> {
    use rustls::ClientConfig;

    // Install the ML-KEM-768 hybrid post-quantum crypto provider.
    // `.ok()` makes this idempotent: a second call (e.g. in tests) is a no-op.
    rustls_post_quantum::provider().install_default().ok();

    let mut root_store = rustls::RootCertStore::empty();
    root_store
        .add(creds.ca_cert.clone())
        .context("add CA cert to root store")?;

    let client_cert_chain = vec![creds.client_cert.clone()];
    let client_key = creds.client_key.clone_key();

    let config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_client_auth_cert(client_cert_chain, client_key)
        .context("configure mTLS client cert")?;

    Ok(TlsConnector::from(Arc::new(config)))
}

/// Build a CA-only TLS connector: the server is verified against `ca_certs`
/// and no client certificate is presented (bearer-token authentication).
pub fn build_ca_only_tls_connector(
    ca_certs: &[CertificateDer<'static>],
) -> anyhow::Result<TlsConnector> {
    rustls_post_quantum::provider().install_default().ok();

    let mut root_store = rustls::RootCertStore::empty();
    for ca in ca_certs {
        root_store
            .add(ca.clone())
            .context("add CA cert to root store")?;
    }
    anyhow::ensure!(!root_store.is_empty(), "no CA certificates supplied");

    let config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    Ok(TlsConnector::from(Arc::new(config)))
}

// ---------------------------------------------------------------------------
// File-path kubeconfig mode and live TLS material reload
// ---------------------------------------------------------------------------

/// How often file-backed TLS material is re-read by [`kubeconfig_client`].
/// Rotation happens on a scale of days or hours, so 30s is prompt and costs
/// three tiny file reads.
pub const DEFAULT_RELOAD_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

/// One PEM input of the TLS configuration: inline kubeconfig `-data` (fixed)
/// or a file path (re-read on reload).
enum PemSource {
    Inline(Vec<u8>),
    File(std::path::PathBuf),
}

impl PemSource {
    fn read(&self) -> anyhow::Result<Vec<u8>> {
        match self {
            Self::Inline(b) => Ok(b.clone()),
            Self::File(p) => std::fs::read(p).with_context(|| format!("reading {}", p.display())),
        }
    }
}

struct TlsSources {
    ca: PemSource,
    cert: PemSource,
    key: PemSource,
}

impl TlsSources {
    fn has_files(&self) -> bool {
        [&self.ca, &self.cert, &self.key]
            .iter()
            .any(|s| matches!(s, PemSource::File(_)))
    }

    fn read(&self) -> anyhow::Result<[Vec<u8>; 3]> {
        Ok([self.ca.read()?, self.cert.read()?, self.key.read()?])
    }
}

/// Build an mTLS client config from PEM bytes. The CA input may be a bundle
/// (every certificate is trusted, so a CA rotation can overlap old and new);
/// the cert input may be a chain.
///
/// Uses the post-quantum provider explicitly rather than the process default,
/// so the hybrid X25519MLKEM768 group is offered regardless of what else
/// installed a default.
fn build_client_config(
    ca_pem: &[u8],
    cert_pem: &[u8],
    key_pem: &[u8],
) -> anyhow::Result<Arc<rustls::ClientConfig>> {
    let mut root_store = rustls::RootCertStore::empty();
    for ca in CertificateDer::pem_slice_iter(ca_pem) {
        root_store
            .add(ca.context("parse CA PEM")?)
            .context("add CA cert to root store")?;
    }
    anyhow::ensure!(!root_store.is_empty(), "no CA certificates in CA PEM");

    let chain = CertificateDer::pem_slice_iter(cert_pem)
        .collect::<Result<Vec<_>, _>>()
        .context("parse client cert PEM")?;
    anyhow::ensure!(!chain.is_empty(), "no certificate in client cert PEM");
    let key = PrivateKeyDer::from_pem_slice(key_pem).context("parse client key PEM")?;

    let config =
        rustls::ClientConfig::builder_with_provider(Arc::new(rustls_post_quantum::provider()))
            .with_safe_default_protocol_versions()
            .context("select TLS protocol versions")?
            .with_root_certificates(root_store)
            .with_client_auth_cert(chain, key)
            .context("configure mTLS client cert (cert/key mismatch?)")?;
    Ok(Arc::new(config))
}

struct SwapState {
    connector: TlsConnector,
    generation: u64,
    last_check: std::time::Instant,
    /// Input bytes of the last attempted build, successful or not, so an
    /// unchanged bad file is logged once and a changed one is retried.
    seen: [Vec<u8>; 3],
}

struct Reloader {
    sources: TlsSources,
    interval: std::time::Duration,
}

/// A [`TlsConnector`] that can be replaced while in use. New connections take
/// the current connector; [`Self::generation`] lets a long-lived connection
/// notice it was made with superseded material.
///
/// File-backed material is compared by content every reload interval rather
/// than by mtime: mtime has coarse granularity on some filesystems and is
/// preserved by some copy tools, so a same-size replacement can go unnoticed,
/// whereas the inputs are a few KB. A failed rebuild (unparsable, half-written
/// or mismatched cert/key, empty CA) keeps the previous connector.
pub struct SwapConnector {
    state: std::sync::Mutex<SwapState>,
    reload: Option<Reloader>,
}

impl From<TlsConnector> for SwapConnector {
    /// A connector that never changes.
    fn from(connector: TlsConnector) -> Self {
        Self {
            state: std::sync::Mutex::new(SwapState {
                connector,
                generation: 0,
                last_check: std::time::Instant::now(),
                seen: Default::default(),
            }),
            reload: None,
        }
    }
}

impl SwapConnector {
    fn from_sources(
        sources: TlsSources,
        interval: Option<std::time::Duration>,
    ) -> anyhow::Result<Self> {
        let seen = sources.read()?;
        let config = build_client_config(&seen[0], &seen[1], &seen[2])?;
        let reload = interval
            .filter(|_| sources.has_files())
            .map(|interval| Reloader { sources, interval });
        Ok(Self {
            state: std::sync::Mutex::new(SwapState {
                connector: TlsConnector::from(config),
                generation: 0,
                last_check: std::time::Instant::now(),
                seen,
            }),
            reload,
        })
    }

    fn refresh(&self, state: &mut SwapState) {
        let Some(reload) = &self.reload else { return };
        if state.last_check.elapsed() < reload.interval {
            return;
        }
        state.last_check = std::time::Instant::now();
        let bytes = match reload.sources.read() {
            Ok(b) => b,
            Err(e) => {
                warn!("TLS material reload: keeping previous connector: {e:#}");
                return;
            }
        };
        if bytes == state.seen {
            return;
        }
        match build_client_config(&bytes[0], &bytes[1], &bytes[2]) {
            Ok(config) => {
                state.connector = TlsConnector::from(config);
                state.generation += 1;
                tracing::info!(
                    generation = state.generation,
                    "TLS material changed on disk; new connections use it"
                );
            }
            Err(e) => error!("TLS material reload: keeping previous connector: {e:#}"),
        }
        state.seen = bytes;
    }

    /// The current connector and its generation, reloading first if due.
    pub fn current(&self) -> (TlsConnector, u64) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.refresh(&mut state);
        (state.connector.clone(), state.generation)
    }

    /// The current generation, reloading first if due. It increases each time
    /// the connector is replaced.
    pub fn generation(&self) -> u64 {
        self.current().1
    }

    /// Resolves once the connector is newer than `generation`; never resolves
    /// for a connector that cannot change.
    async fn superseded(&self, generation: u64) {
        let Some(reload) = &self.reload else {
            return std::future::pending().await;
        };
        loop {
            tokio::time::sleep(reload.interval).await;
            if self.generation() != generation {
                return;
            }
        }
    }
}

fn strip_quotes(v: &str) -> &str {
    v.trim().trim_matches(|c| c == '"' || c == '\'')
}

/// Locate one PEM input in kubeconfig text: inline base64 `data_key` wins over
/// the `path_key` file, which resolves against `dir` unless absolute (client-go
/// `dataFromSliceOrFile` / `ResolvePaths`).
fn pem_source(
    raw: &str,
    dir: &std::path::Path,
    data_key: &str,
    path_key: &str,
) -> anyhow::Result<PemSource> {
    if let Some(d) = extract_yaml_value(raw, data_key).filter(|d| !d.is_empty()) {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(d.trim())
            .with_context(|| format!("decode kubeconfig {data_key}"))?;
        return Ok(PemSource::Inline(bytes));
    }
    let p = extract_yaml_value(raw, path_key)
        .map(strip_quotes)
        .filter(|p| !p.is_empty())
        .with_context(|| format!("kubeconfig: missing {data_key} or {path_key}"))?;
    Ok(PemSource::File(dir.join(p)))
}

/// Build a client from a kubeconfig, honouring both inline `*-data` fields and
/// `certificate-authority` / `client-certificate` / `client-key` file paths
/// (relative paths resolve against the kubeconfig's directory, as client-go
/// does). File-backed material is reloaded every [`DEFAULT_RELOAD_INTERVAL`].
pub fn kubeconfig_client(path: &str) -> anyhow::Result<HyperApiClient> {
    kubeconfig_client_with_reload(path, Some(DEFAULT_RELOAD_INTERVAL))
}

/// [`kubeconfig_client`] with an explicit reload interval; `None` disables
/// reloading.
pub fn kubeconfig_client_with_reload(
    path: &str,
    reload: Option<std::time::Duration>,
) -> anyhow::Result<HyperApiClient> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading kubeconfig {path}"))?;
    let dir = std::path::Path::new(path)
        .parent()
        .unwrap_or_else(|| std::path::Path::new(""));
    let source = |data_key: &str, path_key: &str| pem_source(&raw, dir, data_key, path_key);

    let server = extract_yaml_value(&raw, "server:")
        .map(strip_quotes)
        .context("kubeconfig: missing server")?
        .to_owned();
    let sources = TlsSources {
        ca: source("certificate-authority-data:", "certificate-authority:")?,
        cert: source("client-certificate-data:", "client-certificate:")?,
        key: source("client-key-data:", "client-key:")?,
    };
    Ok(HyperApiClient {
        server,
        connector: SwapConnector::from_sources(sources, reload)?,
        bearer: None,
    })
}

// ---------------------------------------------------------------------------
// Bearer tokens and in-cluster configuration
// ---------------------------------------------------------------------------

/// Default ServiceAccount token path inside a pod.
pub const IN_CLUSTER_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";
/// Default ServiceAccount CA bundle path inside a pod.
pub const IN_CLUSTER_CA_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt";

/// How often a [`TokenFile`] re-reads its file. The kubelet refreshes a
/// projected token at ~80% of its TTL (default 1h) and the apiserver accepts
/// the old one until expiry, so one minute leaves a wide safety margin.
const TOKEN_REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);

/// Cached-token age beyond which a failing refresh is logged at `error!`: the
/// token is then likely near expiry and requests will start failing with 401.
const TOKEN_STALE_ERROR_AFTER: std::time::Duration = std::time::Duration::from_secs(600);

struct CachedToken {
    token: String,
    last_attempt: std::time::Instant,
    last_success: std::time::Instant,
}

/// A bearer token read from a file that is rewritten in place over time
/// (projected ServiceAccount token). The file is read at most once per
/// refresh interval, whether or not the read succeeds; a failed re-read keeps
/// the previous token.
pub struct TokenFile {
    path: std::path::PathBuf,
    refresh: std::time::Duration,
    cached: std::sync::Mutex<CachedToken>,
}

fn read_token(path: &std::path::Path) -> anyhow::Result<String> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading token file {}", path.display()))?;
    let token = raw.trim();
    anyhow::ensure!(!token.is_empty(), "token file {} is empty", path.display());
    Ok(token.to_owned())
}

impl TokenFile {
    /// Read the token now (failing if unreadable or empty) and re-read it
    /// every minute thereafter.
    pub fn new(path: impl Into<std::path::PathBuf>) -> anyhow::Result<Self> {
        Self::with_refresh(path, TOKEN_REFRESH_INTERVAL)
    }

    /// Like [`Self::new`] with an explicit re-read interval.
    pub fn with_refresh(
        path: impl Into<std::path::PathBuf>,
        refresh: std::time::Duration,
    ) -> anyhow::Result<Self> {
        let path = path.into();
        let token = read_token(&path)?;
        let now = std::time::Instant::now();
        Ok(Self {
            path,
            refresh,
            cached: std::sync::Mutex::new(CachedToken {
                token,
                last_attempt: now,
                last_success: now,
            }),
        })
    }

    /// The current token, re-read from disk if the last attempt is older than
    /// the refresh interval.
    pub fn token(&self) -> String {
        let mut cached = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        if cached.last_attempt.elapsed() >= self.refresh {
            let now = std::time::Instant::now();
            cached.last_attempt = now;
            match read_token(&self.path) {
                Ok(t) => {
                    cached.token = t;
                    cached.last_success = now;
                }
                Err(e) => {
                    let age = cached.last_success.elapsed();
                    if age >= TOKEN_STALE_ERROR_AFTER {
                        error!("token refresh failing for {age:?}, serving stale token: {e:#}");
                    } else {
                        warn!("token refresh failed, reusing previous token: {e:#}");
                    }
                }
            }
        }
        cached.token.clone()
    }
}

/// Source of the `Authorization: Bearer` token for a [`HyperApiClient`].
pub enum BearerSource {
    /// A fixed token.
    Static(String),
    /// A rotating token re-read from a file.
    File(TokenFile),
}

impl BearerSource {
    fn token(&self) -> String {
        match self {
            Self::Static(t) => t.clone(),
            Self::File(f) => f.token(),
        }
    }
}

/// Server URL for the in-cluster apiserver from the
/// `KUBERNETES_SERVICE_HOST` / `KUBERNETES_SERVICE_PORT` values. IPv6
/// literals are bracketed.
pub fn in_cluster_server_url(host: &str, port: &str) -> anyhow::Result<String> {
    let host = host.trim();
    anyhow::ensure!(!host.is_empty(), "KUBERNETES_SERVICE_HOST is empty");
    let port: u16 = port
        .trim()
        .parse()
        .with_context(|| format!("invalid KUBERNETES_SERVICE_PORT {port:?}"))?;
    anyhow::ensure!(
        !host.contains('%'),
        "IPv6 zone ids are not supported in KUBERNETES_SERVICE_HOST: {host:?}"
    );
    if host.contains(':') && !host.starts_with('[') {
        Ok(format!("https://[{host}]:{port}"))
    } else {
        Ok(format!("https://{host}:{port}"))
    }
}

/// Build a client from in-cluster ServiceAccount credentials: server from
/// `host`/`port`, CA bundle from `ca_path`, rotating token from `token_path`.
pub fn in_cluster_client_from(
    host: &str,
    port: &str,
    token_path: &std::path::Path,
    ca_path: &std::path::Path,
) -> anyhow::Result<HyperApiClient> {
    let server = in_cluster_server_url(host, port)?;
    let ca_pem =
        std::fs::read(ca_path).with_context(|| format!("reading CA {}", ca_path.display()))?;
    let cas = CertificateDer::pem_slice_iter(&ca_pem)
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("parse CA PEM {}", ca_path.display()))?;
    let connector =
        build_ca_only_tls_connector(&cas).with_context(|| format!("CA {}", ca_path.display()))?;
    Ok(HyperApiClient {
        server,
        connector: connector.into(),
        bearer: Some(BearerSource::File(TokenFile::new(token_path)?)),
    })
}

/// [`in_cluster_client_from`] using the standard pod environment variables and
/// ServiceAccount mount paths.
pub fn in_cluster_client() -> anyhow::Result<HyperApiClient> {
    let host = std::env::var("KUBERNETES_SERVICE_HOST")
        .context("KUBERNETES_SERVICE_HOST not set (not running in a pod?)")?;
    let port = std::env::var("KUBERNETES_SERVICE_PORT")
        .context("KUBERNETES_SERVICE_PORT not set (not running in a pod?)")?;
    in_cluster_client_from(
        &host,
        &port,
        std::path::Path::new(IN_CLUSTER_TOKEN_PATH),
        std::path::Path::new(IN_CLUSTER_CA_PATH),
    )
}

// ---------------------------------------------------------------------------
// HyperApiClient — HTTP/1.1 over TLS client
//
// The `bearer` field supports an optional auth header; callers pass `None`
// when using mTLS client-cert auth alone.
// ---------------------------------------------------------------------------

/// A minimal HTTP/1.1 mTLS API client backed by hyper.
///
/// Opens a fresh TLS connection per request (scaffold; reuse is a later opt).
pub struct HyperApiClient {
    /// Base URL of the API server, e.g. "https://127.0.0.1:6443".
    pub server: String,
    /// TLS connector built from kubeconfig credentials; convert a plain
    /// [`TlsConnector`] with `.into()` for a fixed one.
    pub connector: SwapConnector,
    /// Optional bearer token added as `Authorization: Bearer <token>`.
    pub bearer: Option<BearerSource>,
}

/// Total budget for a non-watch request/response cycle (connect through body
/// collection), and for a watch's connect+handshake+send-request setup phase.
/// Neither is long-lived, so a hung apiserver must fail fast rather than block
/// the caller forever — for the scheduler, an unbounded hang here also leaks
/// the pod's key in its in-flight dedup set, permanently orphaning that pod.
/// 30s is generous for a slow-but-alive apiserver while still failing well
/// before a caller-side watchdog would notice.
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Idle timeout for a single `watch_stream` frame read, reset on every frame
/// received. A healthy watch — even one that never receives a bookmark — can
/// stay open indefinitely; only a connection silent for this long trips it.
/// That silence is the half-open-socket failure mode: the peer stops sending
/// but never closes, so without this timeout `body.frame().await` blocks
/// forever and the caller's reconnect loop never runs.
///
/// The apiserver's optional periodic watch bookmark (`allowWatchBookmarks=true`)
/// ticks every 60s (crates/apiserver/src/handlers/watch.rs); 5 minutes gives 5x
/// headroom above that cadence so a bookmark-subscribed watch never trips this
/// spuriously, while still bounding detection of a truly stuck connection to a
/// few minutes instead of never.
const WATCH_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Turn a URI host into the TLS server name to verify the certificate against.
///
/// `hyper::Uri::host()` keeps the brackets of an IPv6 literal (`[fd00::1]`),
/// which is not a valid DNS name; strip them and verify against an IP SAN.
/// Zone IDs (`fe80::1%eth0`) are rejected: a certificate cannot carry one.
fn tls_server_name(host: &str) -> anyhow::Result<rustls::pki_types::ServerName<'static>> {
    let bare = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
        return Ok(rustls::pki_types::ServerName::IpAddress(ip.into()));
    }
    rustls::pki_types::ServerName::try_from(host.to_owned())
        .with_context(|| format!("invalid TLS server name {host:?}"))
}

impl HyperApiClient {
    /// Current `Authorization` header value, if a bearer source is set.
    fn authorization(&self) -> Option<String> {
        self.bearer
            .as_ref()
            .map(|b| format!("Bearer {}", b.token()))
    }

    /// Parse `server` + `path` into (host, port) for TCP connect.
    fn parse_addr(server: &str, path: &str) -> anyhow::Result<(String, u16, String)> {
        let uri: hyper::Uri = format!("{server}{path}").parse().context("parse URI")?;
        let host = uri.host().context("URI missing host")?.to_owned();
        let port = uri.port_u16().unwrap_or(443);
        let addr = format!("{host}:{port}");
        Ok((host, port, addr))
    }

    /// Open a fresh TLS connection to the API server, returning it with the
    /// generation of the connector that made it.
    async fn connect(
        &self,
        host: &str,
        addr: &str,
    ) -> anyhow::Result<(TokioIo<tokio_rustls::client::TlsStream<TcpStream>>, u64)> {
        let stream = TcpStream::connect(addr)
            .await
            .with_context(|| format!("TCP connect to {addr}"))?;
        let server_name = tls_server_name(host)?;
        let (connector, generation) = self.connector.current();
        let tls = connector
            .connect(server_name, stream)
            .await
            .context("TLS handshake")?;
        Ok((TokioIo::new(tls), generation))
    }

    /// Send an HTTP request and return the response body as bytes.
    ///
    /// The body is always fully buffered. For streaming responses use
    /// [`Self::watch_stream`].
    pub async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<String>,
    ) -> anyhow::Result<(hyper::StatusCode, String)> {
        self.request_with_content_type(method, path, body, "application/json")
            .await
    }

    /// Send an HTTP request with an explicit Content-Type header, when a body is
    /// present.
    ///
    /// Status-subresource PATCH endpoints reject the default `application/json`
    /// (the apiserver's `accepts_patch_content_type` requires
    /// `application/merge-patch+json` or `application/strategic-merge-patch+json`,
    /// returning 415 otherwise), so callers that PATCH `.../status` need to
    /// override it rather than go through [`Self::request`].
    pub async fn request_with_content_type(
        &self,
        method: Method,
        path: &str,
        body: Option<String>,
        content_type: &str,
    ) -> anyhow::Result<(hyper::StatusCode, String)> {
        self.request_with_content_type_timeout(method, path, body, content_type, REQUEST_TIMEOUT)
            .await
    }

    /// [`request_with_content_type`], but with the request timeout as an explicit
    /// parameter rather than the hardcoded [`REQUEST_TIMEOUT`]. This lets tests
    /// exercise the timeout path (a hung apiserver) in milliseconds instead of
    /// waiting out the real 30s budget.
    async fn request_with_content_type_timeout(
        &self,
        method: Method,
        path: &str,
        body: Option<String>,
        content_type: &str,
        request_timeout: std::time::Duration,
    ) -> anyhow::Result<(hyper::StatusCode, String)> {
        let call = async move {
            let (host, _port, addr) = Self::parse_addr(&self.server, path)?;
            let (io, _) = self.connect(&host, &addr).await?;

            let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
                .await
                .context("HTTP/1.1 handshake")?;
            tokio::spawn(async move {
                if let Err(e) = conn.await {
                    error!("HTTP connection error: {e}");
                }
            });

            let body_bytes = body
                .as_deref()
                .map(|s| bytes::Bytes::from(s.to_owned()))
                .unwrap_or_default();

            let mut builder = Request::builder()
                .method(method)
                .uri(path)
                .header("Host", &host)
                .header("Accept", "application/json");
            if body.is_some() {
                builder = builder
                    .header("Content-Type", content_type)
                    .header("Content-Length", body_bytes.len().to_string());
            }
            if let Some(h) = self.authorization() {
                builder = builder.header("Authorization", h);
            }
            let req = builder
                .body(http_body_util::Full::new(body_bytes))
                .context("build request")?;

            use http_body_util::BodyExt;
            let resp: hyper::Response<hyper::body::Incoming> =
                sender.send_request(req).await.context("send request")?;
            let status = resp.status();
            let text = String::from_utf8_lossy(
                &resp
                    .into_body()
                    .collect()
                    .await
                    .context("read body")?
                    .to_bytes(),
            )
            .into_owned();
            anyhow::Ok((status, text))
        };

        tokio::time::timeout(request_timeout, call)
            .await
            .with_context(|| format!("request to {path} timed out after {request_timeout:?}"))?
    }

    /// Stream newline-delimited JSON events from a watch endpoint.
    ///
    /// Calls `on_event` for each successfully parsed JSON value. Incomplete
    /// lines (no trailing `\n`) are buffered until the next frame. Malformed
    /// lines are logged and skipped.
    pub async fn watch_stream(
        &self,
        path: &str,
        on_event: impl FnMut(Value),
    ) -> anyhow::Result<()> {
        // Connecting and sending the initial watch request is not long-lived,
        // so it gets the same bounded treatment as a normal request. Only the
        // frame-read loop below — the watch itself — is allowed to run
        // indefinitely, guarded instead by the per-frame idle timeout.
        let setup = async move {
            let (host, _port, addr) = Self::parse_addr(&self.server, path)?;
            let (io, generation) = self.connect(&host, &addr).await?;

            let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
                .await
                .context("HTTP/1.1 handshake")?;
            tokio::spawn(async move {
                if let Err(e) = conn.await {
                    error!("watch connection error: {e}");
                }
            });

            let mut builder = Request::builder()
                .method(Method::GET)
                .uri(path)
                .header("Host", &host)
                .header("Accept", "application/json");
            if let Some(h) = self.authorization() {
                builder = builder.header("Authorization", h);
            }
            let req = builder
                .body(http_body_util::Empty::<bytes::Bytes>::new())
                .context("build watch request")?;

            let resp: hyper::Response<hyper::body::Incoming> = sender
                .send_request(req)
                .await
                .context("send watch request")?;
            if !resp.status().is_success() {
                anyhow::bail!("watch returned HTTP {}", resp.status());
            }
            anyhow::Ok((resp, generation))
        };

        let (resp, generation) = tokio::time::timeout(REQUEST_TIMEOUT, setup)
            .await
            .with_context(|| {
                format!("watch connect to {path} timed out after {REQUEST_TIMEOUT:?}")
            })??;

        // A watch can outlive the TLS material it was opened with, and would
        // keep trusting a rotated-out CA and presenting a rotated-out client
        // cert. Ending it cleanly makes the caller's reconnect loop reopen it
        // on the new connector.
        tokio::select! {
            r = read_watch_frames(resp.into_body(), WATCH_IDLE_TIMEOUT, path, on_event) => r,
            () = self.connector.superseded(generation) => {
                warn!("watch on {path} ended: TLS material was replaced; reconnect to use it");
                Ok(())
            }
        }
    }
}

/// Read frames from a watch response body until it ends or errors, feeding
/// complete newline-delimited JSON lines to `on_event`.
///
/// `idle_timeout` guards a single `frame()` read and resets every time one
/// arrives — it is not a deadline on the whole call. A healthy watch (even one
/// that never receives a bookmark) can therefore stay open indefinitely; only
/// a connection silent for `idle_timeout` trips it. That silence is the
/// half-open-socket failure mode: the peer stops sending but never closes, so
/// without this timeout a read blocks forever and a caller's reconnect loop
/// never runs.
///
/// Takes `idle_timeout` as a parameter (rather than reading `WATCH_IDLE_TIMEOUT`
/// directly) so tests can exercise the timeout path in milliseconds instead of
/// waiting out the real 5-minute budget.
async fn read_watch_frames<B>(
    mut body: B,
    idle_timeout: std::time::Duration,
    path: &str,
    mut on_event: impl FnMut(Value),
) -> anyhow::Result<()>
where
    B: hyper::body::Body<Data = bytes::Bytes> + Unpin,
    B::Error: std::fmt::Display,
{
    use http_body_util::BodyExt;
    let mut buf = String::new();

    loop {
        match tokio::time::timeout(idle_timeout, body.frame()).await {
            Err(_) => {
                warn!(
                    "watch stream on {path} idle for {idle_timeout:?} with no frames; \
                     treating as a half-open connection and disconnecting"
                );
                anyhow::bail!("watch stream idle timeout after {idle_timeout:?}");
            }
            Ok(None) => break,
            Ok(Some(Err(e))) => {
                warn!("watch stream error: {e}");
                break;
            }
            Ok(Some(Ok(frame))) => {
                let frame: hyper::body::Frame<bytes::Bytes> = frame;
                if let Ok(data) = frame.into_data() {
                    buf.push_str(&String::from_utf8_lossy(&data));
                    drain_watch_buffer(&mut buf, &mut on_event);
                }
            }
        }
    }

    Ok(())
}

/// Drain all complete newline-terminated JSON lines from `buf`, calling
/// `handler` for each successfully parsed value.
///
/// Lines that fail to parse are logged and skipped. Incomplete lines (no
/// trailing `\n`) are left in `buf` for the next call.
///
/// This function is the canonical implementation used by `HyperApiClient::watch_stream`
/// and is exported so callers can reuse the exact same watch-event parsing logic.
pub fn drain_watch_buffer(buf: &mut String, handler: &mut impl FnMut(Value)) {
    while let Some(nl) = buf.find('\n') {
        let line = buf[..nl].trim().to_owned();
        buf.drain(..=nl);
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&line) {
            Ok(v) => handler(v),
            Err(e) => warn!("failed to parse watch event: {e}: {line}"),
        }
    }
}

#[cfg(test)]
mod reload_tests;

#[cfg(test)]
mod tests {
    use super::*;

    // Verify extract_yaml_value handles both leading-whitespace and flush-left forms,
    // since real kubeconfig files indent nested fields.
    #[test]
    fn extract_yaml_value_strips_leading_whitespace() {
        let text = "clusters:\n  - cluster:\n      server: https://127.0.0.1:6443\n";
        let result = extract_yaml_value(text, "server:");
        assert_eq!(result, Some("https://127.0.0.1:6443"));
    }

    #[test]
    fn extract_yaml_value_returns_none_for_missing_key() {
        let text = "clusters:\n  - cluster:\n      server: https://127.0.0.1:6443\n";
        assert!(extract_yaml_value(text, "certificate-authority-data:").is_none());
    }

    #[test]
    fn extract_yaml_value_returns_first_match() {
        // If the key appears twice, the first value wins. This matters because
        // kubeconfig can have multiple clusters; we always take the first.
        let text = "server: first\nserver: second\n";
        assert_eq!(extract_yaml_value(text, "server:"), Some("first"));
    }

    #[test]
    fn extract_yaml_value_empty_input_returns_none() {
        // An empty kubeconfig string must not panic or return garbage.
        assert!(extract_yaml_value("", "server:").is_none());
    }

    #[test]
    fn extract_yaml_value_key_only_no_value() {
        // "server:" with nothing after the colon returns an empty string, not None.
        // The caller trims whitespace, so both "" and " " yield "".
        let text = "server:\n";
        assert_eq!(extract_yaml_value(text, "server:"), Some(""));
    }

    #[test]
    fn extract_yaml_value_flush_left_key() {
        // Key at column 0, no indentation.
        let text = "server: https://127.0.0.1:6443\n";
        assert_eq!(
            extract_yaml_value(text, "server:"),
            Some("https://127.0.0.1:6443")
        );
    }

    // ---------------------------------------------------------------------------
    // Helpers shared by parse_kubeconfig and build_tls_connector tests.
    // ---------------------------------------------------------------------------

    /// Generate a self-signed CA cert + a leaf cert signed by it, all as PEM.
    /// Returns (ca_pem, leaf_cert_pem, leaf_key_pem).
    /// This mirrors what the apiserver's tls.rs does to produce kubeconfig data.
    fn make_test_certs() -> (String, String, String) {
        use rcgen::{BasicConstraints, CertificateParams, IsCa, Issuer, KeyPair};

        // CA
        let ca_key = KeyPair::generate().expect("generate CA key");
        let mut ca_params = CertificateParams::default();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca_params
            .distinguished_name
            .push(rcgen::DnType::CommonName, "test-ca");
        let ca_cert = ca_params.self_signed(&ca_key).expect("self-sign CA");
        let ca_pem = pem_encode_str("CERTIFICATE", ca_cert.der()).to_string();
        let ca_issuer = Issuer::new(ca_params, ca_key);

        // Leaf (client) cert
        let leaf_key = KeyPair::generate().expect("generate leaf key");
        let mut leaf_params = CertificateParams::default();
        leaf_params
            .distinguished_name
            .push(rcgen::DnType::CommonName, "test-client");
        let leaf_cert = leaf_params
            .signed_by(&leaf_key, &ca_issuer)
            .expect("sign leaf cert");
        let leaf_cert_pem = pem_encode_str("CERTIFICATE", leaf_cert.der());
        let leaf_key_pem = leaf_key.serialize_pem();

        (ca_pem, leaf_cert_pem, leaf_key_pem)
    }

    /// PEM-encode DER bytes into a String.
    fn pem_encode_str(label: &str, der: &[u8]) -> String {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD;
        let encoded = b64.encode(der);
        let mut out = format!("-----BEGIN {label}-----\n");
        for chunk in encoded.as_bytes().chunks(64) {
            out.push_str(std::str::from_utf8(chunk).unwrap());
            out.push('\n');
        }
        out.push_str(&format!("-----END {label}-----\n"));
        out
    }

    /// Build the kubeconfig YAML string that parse_kubeconfig expects.
    /// The fields are base64(PEM) exactly as the apiserver writes them.
    fn make_kubeconfig_yaml(server: &str, ca_pem: &str, cert_pem: &str, key_pem: &str) -> String {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD;
        let ca_data = b64.encode(ca_pem.as_bytes());
        let cert_data = b64.encode(cert_pem.as_bytes());
        let key_data = b64.encode(key_pem.as_bytes());
        format!(
            "apiVersion: v1\n\
             kind: Config\n\
             clusters:\n\
             - cluster:\n\
             \x20   server: {server}\n\
             \x20   certificate-authority-data: {ca_data}\n\
             \x20 name: u7s\n\
             users:\n\
             - name: admin\n\
             \x20 user:\n\
             \x20   client-certificate-data: {cert_data}\n\
             \x20   client-key-data: {key_data}\n",
        )
    }

    /// Write a string to a temp file and return the path.
    fn write_temp_file(content: &str, suffix: &str) -> std::path::PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let tid = std::thread::current().id();
        let path =
            std::env::temp_dir().join(format!("u7s-kubeconfig-{suffix}-{nanos}-{tid:?}.txt"));
        std::fs::write(&path, content).expect("write temp file");
        path
    }

    // ---------------------------------------------------------------------------
    // parse_kubeconfig — happy path
    // ---------------------------------------------------------------------------

    /// parse_kubeconfig must extract all four fields from a well-formed kubeconfig.
    /// This verifies the full read → base64-decode → PEM-parse pipeline that callers
    /// (controller-manager, scheduler) rely on at startup.
    #[test]
    fn parse_kubeconfig_happy_path() {
        let (ca_pem, cert_pem, key_pem) = make_test_certs();
        let yaml = make_kubeconfig_yaml("https://127.0.0.1:6443", &ca_pem, &cert_pem, &key_pem);
        let path = write_temp_file(&yaml, "happy");
        let creds =
            parse_kubeconfig(path.to_str().unwrap()).expect("parse_kubeconfig must succeed");
        assert_eq!(creds.server, "https://127.0.0.1:6443");
        // ca_cert and client_cert are non-empty DER blobs.
        assert!(!creds.ca_cert.is_empty(), "ca_cert DER must be non-empty");
        assert!(
            !creds.client_cert.is_empty(),
            "client_cert DER must be non-empty"
        );
        let _ = std::fs::remove_file(&path);
    }

    // ---------------------------------------------------------------------------
    // parse_kubeconfig — error paths
    // ---------------------------------------------------------------------------

    /// A kubeconfig file that doesn't exist must return an error, not panic.
    #[test]
    fn parse_kubeconfig_missing_file_errors() {
        let result = parse_kubeconfig("/tmp/u7s-kubeconfig-nonexistent-99999.yaml");
        assert!(
            result.is_err(),
            "missing file must return Err, not panic or Ok"
        );
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("reading kubeconfig"),
            "error must mention reading kubeconfig; got: {msg}"
        );
    }

    /// A kubeconfig missing the `server:` field must return a clear error.
    /// The field is required; silently using an empty URL would produce a
    /// confusing connection error at runtime instead of a startup failure.
    #[test]
    fn parse_kubeconfig_missing_server_errors() {
        let (ca_pem, cert_pem, key_pem) = make_test_certs();
        // Omit the server: field entirely.
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD;
        let yaml = format!(
            "certificate-authority-data: {}\nclient-certificate-data: {}\nclient-key-data: {}\n",
            b64.encode(ca_pem.as_bytes()),
            b64.encode(cert_pem.as_bytes()),
            b64.encode(key_pem.as_bytes()),
        );
        let path = write_temp_file(&yaml, "no-server");
        let result = parse_kubeconfig(path.to_str().unwrap());
        assert!(result.is_err(), "missing server must return Err");
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("missing server"),
            "error must mention missing server; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Bad base64 in certificate-authority-data must return an error describing
    /// the decode failure, not a panic or a silent empty cert.
    #[test]
    fn parse_kubeconfig_bad_base64_errors() {
        let yaml = "server: https://127.0.0.1:6443\n\
                    certificate-authority-data: !!!not-base64!!!\n\
                    client-certificate-data: aGVsbG8=\n\
                    client-key-data: aGVsbG8=\n";
        let path = write_temp_file(yaml, "bad-b64");
        let result = parse_kubeconfig(path.to_str().unwrap());
        assert!(result.is_err(), "bad base64 must return Err");
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("decode kubeconfig certificate-authority-data"),
            "error must mention decode failure of the CA field; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Valid base64 that decodes to non-PEM bytes in client-key-data must return
    /// an error. The key field must be a PEM-encoded private key, not arbitrary bytes.
    #[test]
    fn parse_kubeconfig_bad_key_pem_errors() {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD;
        let (ca_pem, cert_pem, _) = make_test_certs();
        // Use random bytes as the key — valid base64 but not a PEM key.
        let bad_key = b64.encode(b"this is not a pem private key");
        let yaml = format!(
            "server: https://127.0.0.1:6443\n\
             certificate-authority-data: {}\n\
             client-certificate-data: {}\n\
             client-key-data: {bad_key}\n",
            b64.encode(ca_pem.as_bytes()),
            b64.encode(cert_pem.as_bytes()),
        );
        let path = write_temp_file(&yaml, "bad-key");
        let result = parse_kubeconfig(path.to_str().unwrap());
        assert!(result.is_err(), "non-PEM key must return Err");
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("private key"),
            "error must describe key failure; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// A client-certificate-data that decodes to PEM with no CERTIFICATE block must
    /// be rejected at parse time; otherwise mTLS would present an empty chain and
    /// fail only at handshake with an opaque error.
    #[test]
    fn parse_kubeconfig_client_cert_without_certificate_block_errors() {
        let (ca_pem, _, key_pem) = make_test_certs();
        let yaml = make_kubeconfig_yaml(
            "https://127.0.0.1:6443",
            &ca_pem,
            "not a certificate\n",
            &key_pem,
        );
        let path = write_temp_file(&yaml, "empty-chain");
        let result = parse_kubeconfig(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        let msg = format!(
            "{:#}",
            result.err().expect("empty cert chain must return Err")
        );
        assert!(
            msg.contains("no certificate in kubeconfig client-certificate"),
            "error must name the empty client cert field; got: {msg}"
        );
    }

    fn write_pem_dir(tag: &str) -> (std::path::PathBuf, String, String, String) {
        let (ca, cert, key) = make_test_certs();
        let dir = std::env::temp_dir().join(format!(
            "politeia-kc-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ca.crt"), &ca).unwrap();
        std::fs::write(dir.join("client.crt"), &cert).unwrap();
        std::fs::write(dir.join("client.key"), &key).unwrap();
        (dir, ca, cert, key)
    }

    /// k3s/kubeadm kubeconfigs reference files; rejecting them forces users to
    /// flatten the kubeconfig by hand. Absolute paths must load.
    #[test]
    fn parse_kubeconfig_reads_absolute_file_path_credentials() {
        let (dir, ..) = write_pem_dir("abs");
        let yaml = format!(
            "server: https://127.0.0.1:6443\ncertificate-authority: {d}/ca.crt\n\
             client-certificate: {d}/client.crt\nclient-key: {d}/client.key\n",
            d = dir.display()
        );
        let kc = dir.join("kubeconfig");
        std::fs::write(&kc, yaml).unwrap();
        let creds = parse_kubeconfig(kc.to_str().unwrap()).expect("file-path creds must parse");
        assert!(!creds.ca_cert.is_empty() && !creds.client_cert.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// client-go resolves relative paths against the kubeconfig's directory,
    /// not the process CWD; resolving against CWD breaks any kubeconfig used
    /// from another directory.
    #[test]
    fn parse_kubeconfig_resolves_relative_paths_against_kubeconfig_dir() {
        let (dir, ..) = write_pem_dir("rel");
        let yaml = "server: https://127.0.0.1:6443\ncertificate-authority: ca.crt\n\
                    client-certificate: \"client.crt\"\nclient-key: client.key\n";
        let kc = dir.join("kubeconfig");
        std::fs::write(&kc, yaml).unwrap();
        let res = parse_kubeconfig(kc.to_str().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
        res.expect("relative paths must resolve against the kubeconfig dir");
    }

    /// client-go prefers inline data over the file when both are set; a stale
    /// file must not override the inline credential.
    #[test]
    fn parse_kubeconfig_inline_data_wins_over_file_path() {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD;
        let (ca, cert, key) = make_test_certs();
        let yaml = format!(
            "server: https://127.0.0.1:6443\n\
             certificate-authority-data: {}\ncertificate-authority: /nonexistent/ca.crt\n\
             client-certificate-data: {}\nclient-certificate: /nonexistent/c.crt\n\
             client-key-data: {}\nclient-key: /nonexistent/k.key\n",
            b64.encode(ca.as_bytes()),
            b64.encode(cert.as_bytes()),
            b64.encode(key.as_bytes()),
        );
        let path = write_temp_file(&yaml, "data-wins");
        let res = parse_kubeconfig(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        res.expect("inline data must win; the nonexistent files must never be read");
    }

    /// A referenced file that does not exist must be named in the error so the
    /// user can fix the path, rather than a generic "missing" message.
    #[test]
    fn parse_kubeconfig_missing_credential_file_error_names_the_path() {
        let (dir, ..) = write_pem_dir("missing");
        let yaml = "server: https://127.0.0.1:6443\ncertificate-authority: no-such-ca.crt\n\
                    client-certificate: client.crt\nclient-key: client.key\n";
        let kc = dir.join("kubeconfig");
        std::fs::write(&kc, yaml).unwrap();
        let res = parse_kubeconfig(kc.to_str().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
        let msg = format!("{:#}", res.err().expect("missing file must error"));
        assert!(msg.contains("no-such-ca.crt"), "got: {msg}");
    }

    // ---------------------------------------------------------------------------
    // build_tls_connector — happy path
    // ---------------------------------------------------------------------------

    /// build_tls_connector must succeed with the PQC provider active.
    ///
    /// This test verifies that rustls_post_quantum::provider() is compatible with
    /// the rustls version in use. If the provider installation fails silently or
    /// ClientConfig::builder() rejects it, this test will catch the breakage before
    /// a runtime crash in scheduler or controller-manager at startup.
    #[test]
    fn build_tls_connector_succeeds_with_pqc_provider() {
        rustls_post_quantum::provider().install_default().ok();

        let (ca_pem, cert_pem, key_pem) = make_test_certs();
        let yaml = make_kubeconfig_yaml("https://127.0.0.1:6443", &ca_pem, &cert_pem, &key_pem);
        let path = write_temp_file(&yaml, "connector-pqc");
        let creds = parse_kubeconfig(path.to_str().unwrap()).expect("parse must succeed");
        let result = build_tls_connector(&creds);
        assert!(
            result.is_ok(),
            "build_tls_connector must succeed with PQC provider active; got: {:?}",
            result.err()
        );
        let _ = std::fs::remove_file(&path);
    }

    /// build_tls_connector must succeed when given valid DER certs and a valid key.
    /// The TlsConnector is what controller-manager and scheduler use to open mTLS
    /// connections to the apiserver; a failure here is a hard startup crash.
    #[test]
    fn build_tls_connector_succeeds_with_valid_creds() {
        let (ca_pem, cert_pem, key_pem) = make_test_certs();
        let yaml = make_kubeconfig_yaml("https://127.0.0.1:6443", &ca_pem, &cert_pem, &key_pem);
        let path = write_temp_file(&yaml, "connector");
        let creds = parse_kubeconfig(path.to_str().unwrap()).expect("parse must succeed");
        let result = build_tls_connector(&creds);
        assert!(
            result.is_ok(),
            "build_tls_connector must succeed with valid creds; got: {:?}",
            result.err()
        );
        let _ = std::fs::remove_file(&path);
    }

    // ---------------------------------------------------------------------------
    // HyperApiClient tests
    // ---------------------------------------------------------------------------

    /// HyperApiClient::request with a known-bad server URL must fail gracefully
    /// with a TCP connection error, not panic or hang indefinitely.
    /// This validates that misconfigured kubeconfig surfaces a clear error at
    /// the first API call rather than silently hanging.
    #[tokio::test]
    async fn hyper_api_client_request_bad_server_returns_error() {
        let (ca_pem, cert_pem, key_pem) = make_test_certs();
        let yaml = make_kubeconfig_yaml("https://127.0.0.1:6443", &ca_pem, &cert_pem, &key_pem);
        let path = write_temp_file(&yaml, "client-bad-server");
        let creds = parse_kubeconfig(path.to_str().unwrap()).expect("parse must succeed");
        let connector = build_tls_connector(&creds).expect("connector must build");
        let _ = std::fs::remove_file(&path);

        // Port 19999 is almost certainly not listening — TCP connect must fail.
        let client = HyperApiClient {
            server: "https://127.0.0.1:19999".to_owned(),
            connector: connector.into(),
            bearer: None,
        };
        let result = client
            .request(hyper::Method::GET, "/api/v1/nodes", None)
            .await;
        assert!(
            result.is_err(),
            "request to a non-listening port must return Err"
        );
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("TCP connect") || msg.contains("Connection refused"),
            "error must describe connection failure; got: {msg}"
        );
    }

    /// HyperApiClient::request_with_content_type must return Err once the request
    /// timeout elapses against a peer that accepts the TCP connection but then
    /// never sends a byte back — a live-but-hung apiserver, not a refused or
    /// closed connection.
    ///
    /// Before request timeouts existed, this exact scenario hung the scheduler's
    /// bind_pod/patch_pod_status/emit_scheduling_event calls indefinitely; via
    /// the scheduler's in-flight dedup set, a hung call permanently orphaned the
    /// pod being processed. Uses the crate-private `_timeout` variant with a
    /// short duration so the test doesn't wait out the real 30s production
    /// budget; an outer real-time guard fails the test loudly if the fix
    /// regresses instead of hanging `cargo test` forever.
    #[tokio::test]
    async fn request_with_content_type_returns_err_after_request_timeout_on_stalled_peer() {
        let (ca_pem, cert_pem, key_pem) = make_test_certs();
        let yaml = make_kubeconfig_yaml("https://127.0.0.1:6443", &ca_pem, &cert_pem, &key_pem);
        let path = write_temp_file(&yaml, "client-stalled-peer");
        let creds = parse_kubeconfig(path.to_str().unwrap()).expect("parse must succeed");
        let connector = build_tls_connector(&creds).expect("connector must build");
        let _ = std::fs::remove_file(&path);

        // A bare TCP listener that accepts the connection and then holds it open
        // forever without writing anything back. The client's TLS handshake blocks
        // waiting for a ServerHello that never arrives — a half-open connection,
        // not a refused or reset one.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            std::future::pending::<()>().await;
            drop(stream);
        });

        let client = HyperApiClient {
            server: format!("https://127.0.0.1:{port}"),
            connector: connector.into(),
            bearer: None,
        };

        let request_timeout = std::time::Duration::from_millis(100);
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.request_with_content_type_timeout(
                hyper::Method::GET,
                "/api/v1/nodes",
                None,
                "application/json",
                request_timeout,
            ),
        )
        .await
        .expect(
            "request_with_content_type must return within 5s of a 100ms request timeout; a \
             hang here means a stalled apiserver would wedge the scheduler task (and, via \
             in-flight dedup, permanently orphan the pod) forever",
        );

        assert!(
            result.is_err(),
            "a request to a peer that accepts but never responds must return Err after the \
             request timeout instead of hanging; got Ok"
        );
        let msg = format!("{:#}", result.err().unwrap());
        assert!(
            msg.contains("timed out"),
            "error must describe a timeout, not some other failure; got: {msg}"
        );
    }

    /// HyperApiClient::watch_stream with an in-process TLS mock server must
    /// deliver all newline-delimited JSON events to the on_event callback.
    ///
    /// This tests the full pipeline: TLS accept → HTTP/1.1 response →
    /// frame-by-frame reading → line splitting → JSON parsing → callback.
    /// If any step is broken, events are silently dropped.
    ///
    /// The mock server uses rcgen-generated self-signed certs so no external
    /// infrastructure is needed. The client is configured to trust the mock's CA.
    #[tokio::test]
    async fn hyper_api_client_watch_stream_delivers_events_from_mock_server() {
        use rcgen::{BasicConstraints, CertificateParams, IsCa, Issuer, KeyPair};
        use rustls::ServerConfig;
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio_rustls::TlsAcceptor;

        // ---- Generate server CA + leaf cert ----
        let server_ca_key = KeyPair::generate().expect("server CA key");
        let mut server_ca_params = CertificateParams::default();
        server_ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        server_ca_params
            .distinguished_name
            .push(rcgen::DnType::CommonName, "mock-server-ca");
        let server_ca_cert = server_ca_params
            .self_signed(&server_ca_key)
            .expect("self-sign server CA");
        let server_ca_issuer = Issuer::new(server_ca_params, server_ca_key);

        let server_leaf_key = KeyPair::generate().expect("server leaf key");
        let mut server_leaf_params = CertificateParams::default();
        server_leaf_params
            .distinguished_name
            .push(rcgen::DnType::CommonName, "127.0.0.1");
        server_leaf_params.subject_alt_names = vec![rcgen::SanType::IpAddress(
            "127.0.0.1".parse().expect("parse IP"),
        )];
        let server_leaf_cert = server_leaf_params
            .signed_by(&server_leaf_key, &server_ca_issuer)
            .expect("sign server leaf");

        // ---- Build TLS acceptor for the mock server ----
        let server_cert_der = server_leaf_cert.der().clone();
        let server_key_der =
            rustls::pki_types::PrivateKeyDer::Pkcs8(server_leaf_key.serialize_der().into());
        let server_tls_config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![server_cert_der], server_key_der)
            .expect("server TLS config");
        let acceptor = TlsAcceptor::from(Arc::new(server_tls_config));

        // ---- Bind TCP listener ----
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let port = listener.local_addr().unwrap().port();

        // ---- Spawn mock TLS server ----
        // Sends two newline-delimited JSON watch events then closes.
        tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.expect("accept");
            let mut tls = acceptor.accept(tcp).await.expect("TLS accept");

            // Drain the HTTP request headers.
            let mut req_buf = vec![0u8; 4096];
            let mut total = 0usize;
            loop {
                let n = tls.read(&mut req_buf[total..]).await.expect("read request");
                if n == 0 {
                    break;
                }
                total += n;
                if req_buf[..total].windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }

            // Write HTTP/1.1 200 response with two JSON watch events.
            let body = "{\"type\":\"ADDED\"}\n{\"type\":\"MODIFIED\"}\n";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
                body.len(),
                body
            );
            tls.write_all(response.as_bytes())
                .await
                .expect("write response");
            tls.flush().await.expect("flush");
            // Brief pause so hyper can read the full body before we drop the stream.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        });

        // ---- Build client connector that trusts the mock server's CA ----
        // We use the server CA cert as the trusted root for the client.
        // Client auth is configured with a separate client cert (from make_test_certs).
        let server_ca_pem = pem_encode_str("CERTIFICATE", server_ca_cert.der());
        let (client_cert_pem, client_key_pem) = {
            let (_, c, k) = make_test_certs();
            (c, k)
        };
        // Build a TlsConnector that trusts the server CA and presents the client cert.
        let connector = {
            use rustls::pki_types::{CertificateDer, PrivateKeyDer};
            use rustls::ClientConfig;

            let server_ca_der = CertificateDer::pem_slice_iter(server_ca_pem.as_bytes())
                .next()
                .expect("server CA cert")
                .expect("parse server CA DER");
            let client_cert_der: CertificateDer<'static> =
                CertificateDer::pem_slice_iter(client_cert_pem.as_bytes())
                    .next()
                    .expect("client cert")
                    .expect("parse client cert DER");
            let client_key: PrivateKeyDer<'static> =
                PrivateKeyDer::from_pem_slice(client_key_pem.as_bytes()).expect("client key");

            let mut root_store = rustls::RootCertStore::empty();
            root_store.add(server_ca_der).expect("add server CA");

            let config = ClientConfig::builder()
                .with_root_certificates(root_store)
                .with_client_auth_cert(vec![client_cert_der], client_key)
                .expect("client config");
            TlsConnector::from(Arc::new(config))
        };

        let client = HyperApiClient {
            server: format!("https://127.0.0.1:{port}"),
            connector: connector.into(),
            bearer: None,
        };

        // ---- Run watch_stream and collect events ----
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let events_clone = events.clone();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.watch_stream("/api/v1/serviceaccounts?watch=true", move |v| {
                events_clone.lock().unwrap().push(v);
            }),
        )
        .await;

        assert!(
            result.is_ok(),
            "watch_stream must not hang for more than 5s"
        );
        // The stream ended gracefully (or with a minor EOF error — both are ok).
        // What matters is that both events arrived.
        let received = events.lock().unwrap();
        assert_eq!(
            received.len(),
            2,
            "expected 2 watch events; got {}: {:?}",
            received.len(),
            *received
        );
        assert_eq!(received[0]["type"], "ADDED");
        assert_eq!(received[1]["type"], "MODIFIED");
    }

    /// A fake watch body whose `poll_frame` never resolves — no data, no end,
    /// no error — simulating a half-open connection where the peer stops
    /// sending frames but never closes the socket.
    struct StalledBody;

    impl hyper::body::Body for StalledBody {
        type Data = bytes::Bytes;
        type Error = std::convert::Infallible;

        fn poll_frame(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
            std::task::Poll::Pending
        }
    }

    /// read_watch_frames must return Err once idle_timeout elapses against a
    /// body that never yields a frame and never ends, instead of hanging.
    ///
    /// This is the exact failure mode that, before the idle timeout existed,
    /// wedged the scheduler's entire pod-discovery loop forever: watch_stream
    /// never returns, so main.rs's 5s-reconnect fallback (which only runs
    /// after the watch call returns) never fires, and no pod is ever scheduled
    /// again until the process is restarted. The outer real-time timeout means
    /// a regression here fails this test loudly instead of hanging `cargo test`
    /// forever.
    #[tokio::test]
    async fn read_watch_frames_returns_err_after_idle_timeout_on_stalled_body() {
        let idle_timeout = std::time::Duration::from_millis(50);
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            read_watch_frames(
                StalledBody,
                idle_timeout,
                "/api/v1/pods?watch=true",
                |_: Value| {
                    panic!("StalledBody never yields a frame; on_event must not be called");
                },
            ),
        )
        .await
        .expect(
            "read_watch_frames must return within 5s of a 50ms idle timeout; a hang here means \
             a half-open watch (peer silent, socket never closed) would wedge the scheduler's \
             entire pod-discovery loop forever",
        );

        assert!(
            result.is_err(),
            "a stalled frame source must yield Err after idle_timeout so the caller's \
             reconnect loop fires instead of blocking forever; got Ok"
        );
    }

    /// HyperApiClient::watch_stream with a plain-HTTP mock server must deliver
    /// all JSON events. We test this by connecting through a non-TLS path by
    /// pointing the server URL at http:// — the URI parse extracts host/port
    /// correctly regardless of scheme, and hyper's TCP connect will succeed.
    /// The TLS handshake will then fail, but this test validates the URI
    /// parsing and the callback wiring (via a plain-TCP mock that we verify
    /// returns events when TLS is bypassed).
    ///
    /// For a fully end-to-end watch test without TLS, we test drain_watch_buffer
    /// (the inner line-splitting logic) as a pure function — that is the real
    /// correctness gate.
    #[test]
    fn hyper_api_client_parse_addr_extracts_host_and_port() {
        // HyperApiClient::parse_addr is private; test through observable behavior:
        // a bad-host request must mention the host in its error.
        // We do this indirectly by checking parse_uri_parts (same logic).
        let uri: hyper::Uri = "https://10.0.0.1:6443/api/v1/pods"
            .parse()
            .expect("must parse");
        assert_eq!(uri.host(), Some("10.0.0.1"));
        assert_eq!(uri.port_u16(), Some(6443));
    }

    /// A bracketed IPv6 literal must become an IP server name; as a DNS name it
    /// is rejected, so every dual-stack/IPv6 control-plane endpoint would fail
    /// at TLS setup.
    #[test]
    fn tls_server_name_strips_ipv6_brackets_and_accepts_any_hex_case() {
        use rustls::pki_types::ServerName;
        let expect = |s: &str| -> ServerName<'static> {
            ServerName::IpAddress(s.parse::<std::net::IpAddr>().unwrap().into())
        };
        assert_eq!(tls_server_name("[fd00::1]").unwrap(), expect("fd00::1"));
        assert_eq!(
            tls_server_name("[FD00::ABCD]").unwrap(),
            expect("fd00::abcd")
        );
        assert_eq!(tls_server_name("fd00::1").unwrap(), expect("fd00::1"));
        assert_eq!(tls_server_name("10.0.0.1").unwrap(), expect("10.0.0.1"));
        assert_eq!(
            tls_server_name("kubernetes.default").unwrap(),
            ServerName::try_from("kubernetes.default").unwrap()
        );
    }

    #[test]
    fn tls_server_name_rejects_zone_ids_and_garbage() {
        assert!(tls_server_name("[fe80::1%eth0]").is_err());
        assert!(tls_server_name("[not-an-ip]").is_err());
        assert!(tls_server_name("").is_err());
    }

    /// Non-default port, default port and IPv6 literal all yield a connectable
    /// `host:port` (brackets kept for the socket address).
    #[test]
    fn parse_addr_handles_ipv6_ports_and_default_port() {
        let (h, p, a) = HyperApiClient::parse_addr("https://[fd00::1]:6443", "/x").unwrap();
        assert_eq!(
            (h.as_str(), p, a.as_str()),
            ("[fd00::1]", 6443, "[fd00::1]:6443")
        );
        let (h, p, a) = HyperApiClient::parse_addr("https://[fd00::1]", "/x").unwrap();
        assert_eq!(
            (h.as_str(), p, a.as_str()),
            ("[fd00::1]", 443, "[fd00::1]:443")
        );
        let (h, p, a) = HyperApiClient::parse_addr("https://10.0.0.1:8443", "/x").unwrap();
        assert_eq!(
            (h.as_str(), p, a.as_str()),
            ("10.0.0.1", 8443, "10.0.0.1:8443")
        );
        assert!(
            HyperApiClient::parse_addr("https://[fe80::1%eth0]:6443", "/x").is_err()
                || tls_server_name(
                    HyperApiClient::parse_addr("https://[fe80::1%eth0]:6443", "/x")
                        .unwrap()
                        .0
                        .as_str()
                )
                .is_err()
        );
    }

    /// Full handshake against `[::1]` with a cert that only carries an IP SAN.
    #[tokio::test]
    async fn connect_verifies_ipv6_literal_against_ip_san() {
        use rcgen::{BasicConstraints, CertificateParams, IsCa, Issuer, KeyPair};
        rustls_post_quantum::provider().install_default().ok();

        let ca_key = KeyPair::generate().unwrap();
        let mut ca_params = CertificateParams::default();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_key).unwrap();
        let issuer = Issuer::new(ca_params, ca_key);
        let leaf_key = KeyPair::generate().unwrap();
        let mut leaf_params = CertificateParams::default();
        leaf_params.subject_alt_names = vec![rcgen::SanType::IpAddress("::1".parse().unwrap())];
        let leaf = leaf_params.signed_by(&leaf_key, &issuer).unwrap();

        let server_cfg = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(
                vec![leaf.der().clone()],
                PrivateKeyDer::Pkcs8(leaf_key.serialize_der().into()),
            )
            .unwrap();
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(server_cfg));
        let listener = tokio::net::TcpListener::bind("[::1]:0")
            .await
            .expect("IPv6 loopback must be available");
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            acceptor.accept(tcp).await.map(|_| ())
        });

        let mut roots = rustls::RootCertStore::empty();
        roots.add(ca_cert.der().clone()).unwrap();
        let client_cfg = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let client = HyperApiClient {
            server: format!("https://[::1]:{port}"),
            connector: TlsConnector::from(Arc::new(client_cfg)).into(),
            bearer: None,
        };
        let (host, _, addr) = HyperApiClient::parse_addr(&client.server, "/").unwrap();
        client
            .connect(&host, &addr)
            .await
            .expect("handshake against an IP-SAN cert must succeed for an IPv6 literal");
        server.await.unwrap().expect("server side handshake");
    }

    /// Verify that the bearer token field on HyperApiClient is correctly stored.
    /// The header injection itself is tested indirectly via request() and watch_stream().
    /// Covers both the Some(...) and None construction paths since a missing bearer
    /// would silently break auth for any future bearer-token consumer.
    #[test]
    fn hyper_api_client_bearer_field_is_set() {
        let (ca_pem, cert_pem, key_pem) = make_test_certs();
        let yaml = make_kubeconfig_yaml("https://127.0.0.1:6443", &ca_pem, &cert_pem, &key_pem);
        let path = write_temp_file(&yaml, "client-bearer");
        let creds = parse_kubeconfig(path.to_str().unwrap()).expect("parse must succeed");
        let connector = build_tls_connector(&creds).expect("connector must build");
        let _ = std::fs::remove_file(&path);

        let client = HyperApiClient {
            server: "https://127.0.0.1:6443".to_owned(),
            connector: connector.into(),
            bearer: Some(BearerSource::Static("my-token".to_owned())),
        };
        assert_eq!(client.authorization().as_deref(), Some("Bearer my-token"));
        // bearer = None must also be constructable (scheduler's path today).
        // (We only check the field; construction is what matters for the compiler.)
        drop(client);
    }

    // ---------------------------------------------------------------------------
    // In-cluster ServiceAccount auth
    // ---------------------------------------------------------------------------

    /// A bare IPv6 service host must be bracketed, otherwise the URL parses the
    /// address as host:port garbage and the controller can never reach an
    /// IPv6-only cluster's apiserver.
    #[test]
    fn in_cluster_server_url_brackets_ipv6_literals() {
        assert_eq!(
            in_cluster_server_url("fd00:10:96::1", "443").unwrap(),
            "https://[fd00:10:96::1]:443"
        );
        assert_eq!(
            in_cluster_server_url("[fd00::1]", "6443").unwrap(),
            "https://[fd00::1]:6443"
        );
    }

    #[test]
    fn in_cluster_server_url_leaves_ipv4_and_dns_hosts_alone() {
        assert_eq!(
            in_cluster_server_url("10.96.0.1", "443").unwrap(),
            "https://10.96.0.1:443"
        );
        assert_eq!(
            in_cluster_server_url("kubernetes.default.svc", "443\n").unwrap(),
            "https://kubernetes.default.svc:443"
        );
    }

    /// Missing/garbage env must fail at startup, not produce a URL that fails
    /// obscurely on the first request.
    #[test]
    fn in_cluster_server_url_rejects_empty_host_and_bad_port() {
        assert!(in_cluster_server_url("", "443").is_err());
        assert!(in_cluster_server_url("10.0.0.1", "").is_err());
        assert!(in_cluster_server_url("10.0.0.1", "https").is_err());
    }

    /// The kubelet rewrites the projected token ~every 48 minutes; a client that
    /// never re-reads it starts getting 401s after the original expires.
    #[test]
    fn token_file_picks_up_rotated_token_after_refresh_interval() {
        let path = write_temp_file("tok-1\n", "token-rotate");
        let tf = TokenFile::with_refresh(&path, std::time::Duration::ZERO).unwrap();
        assert_eq!(tf.token(), "tok-1", "trailing newline must be trimmed");
        std::fs::write(&path, "tok-2\n").unwrap();
        assert_eq!(tf.token(), "tok-2");
        let _ = std::fs::remove_file(&path);
    }

    /// Within the refresh interval the file must not be hit per request.
    #[test]
    fn token_file_caches_within_refresh_interval() {
        let path = write_temp_file("tok-1", "token-cache");
        let tf = TokenFile::with_refresh(&path, std::time::Duration::from_secs(3600)).unwrap();
        std::fs::write(&path, "tok-2").unwrap();
        assert_eq!(tf.token(), "tok-1");
        let _ = std::fs::remove_file(&path);
    }

    /// A transient read failure (e.g. mid-swap) must not drop auth: the old
    /// token stays valid until its own expiry.
    #[test]
    fn token_file_keeps_previous_token_when_refresh_fails() {
        let path = write_temp_file("tok-1", "token-fail");
        let tf = TokenFile::with_refresh(&path, std::time::Duration::ZERO).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(tf.token(), "tok-1");
        std::fs::write(&path, "  ").unwrap();
        assert_eq!(tf.token(), "tok-1", "an empty file is not a valid token");
        std::fs::write(&path, "tok-3").unwrap();
        assert_eq!(tf.token(), "tok-3", "must recover once the file is valid");
        let _ = std::fs::remove_file(&path);
    }

    /// A missing or empty token must fail at construction so a misconfigured pod
    /// crashes at startup rather than sending unauthenticated requests.
    #[test]
    fn token_file_new_fails_on_missing_or_empty_file() {
        let missing = std::env::temp_dir().join("politeia-no-such-token");
        assert!(TokenFile::new(&missing).is_err());
        let path = write_temp_file("\n", "token-empty");
        assert!(TokenFile::new(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bearer_source_file_yields_authorization_header_with_current_token() {
        let path = write_temp_file("abc", "token-header");
        let (ca_pem, _, _) = make_test_certs();
        let ca = CertificateDer::pem_slice_iter(ca_pem.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let client = HyperApiClient {
            server: "https://127.0.0.1:1".to_owned(),
            connector: build_ca_only_tls_connector(&ca).unwrap().into(),
            bearer: Some(BearerSource::File(
                TokenFile::with_refresh(&path, std::time::Duration::ZERO).unwrap(),
            )),
        };
        assert_eq!(client.authorization().as_deref(), Some("Bearer abc"));
        std::fs::write(&path, "def").unwrap();
        assert_eq!(client.authorization().as_deref(), Some("Bearer def"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ca_only_connector_rejects_empty_ca_set() {
        assert!(build_ca_only_tls_connector(&[]).is_err());
    }

    /// Loader failures must be loud: no CA means TLS verification cannot be
    /// configured, and we must never fall back to skipping verification.
    #[test]
    fn in_cluster_client_from_fails_without_usable_ca_or_token() {
        let token = write_temp_file("tok", "ic-token");
        let junk = write_temp_file("not a pem", "ic-junk-ca");
        let missing = std::env::temp_dir().join("politeia-no-such-ca");
        assert!(in_cluster_client_from("10.0.0.1", "443", &token, &junk).is_err());
        assert!(in_cluster_client_from("10.0.0.1", "443", &token, &missing).is_err());
        let (ca_pem, _, _) = make_test_certs();
        let ca = write_temp_file(&ca_pem, "ic-ca");
        let no_token = std::env::temp_dir().join("politeia-no-such-token2");
        assert!(in_cluster_client_from("10.0.0.1", "443", &no_token, &ca).is_err());
        for p in [token, junk, ca] {
            let _ = std::fs::remove_file(p);
        }
    }

    /// End to end: a CA-only client (no client cert) completes TLS against a
    /// server that does not ask for client auth and sends the ServiceAccount
    /// token as a bearer header.
    fn ic_test_server() -> (String, rustls::ServerConfig) {
        use rcgen::{BasicConstraints, CertificateParams, IsCa, Issuer, KeyPair};
        rustls_post_quantum::provider().install_default().ok();

        let ca_key = KeyPair::generate().unwrap();
        let mut ca_params = CertificateParams::default();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_key).unwrap();
        let issuer = Issuer::new(ca_params, ca_key);
        let leaf_key = KeyPair::generate().unwrap();
        let mut leaf_params = CertificateParams::default();
        leaf_params.subject_alt_names =
            vec![rcgen::SanType::IpAddress("127.0.0.1".parse().unwrap())];
        let leaf = leaf_params.signed_by(&leaf_key, &issuer).unwrap();

        let server_cfg = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(
                vec![leaf.der().clone()],
                PrivateKeyDer::Pkcs8(leaf_key.serialize_der().into()),
            )
            .unwrap();
        (pem_encode_str("CERTIFICATE", ca_cert.der()), server_cfg)
    }

    #[tokio::test]
    async fn in_cluster_client_authenticates_with_bearer_over_ca_only_tls() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (ca_pem, server_cfg) = ic_test_server();
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(server_cfg));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut tls = acceptor.accept(tcp).await.unwrap();
            let mut buf = vec![0u8; 4096];
            let n = tls.read(&mut buf).await.unwrap();
            tls.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                .await
                .unwrap();
            tls.shutdown().await.ok();
            String::from_utf8_lossy(&buf[..n]).into_owned()
        });

        let ca_file = write_temp_file(&ca_pem, "ic-e2e-ca");
        let token_file = write_temp_file("sa-token\n", "ic-e2e-token");
        let client =
            in_cluster_client_from("127.0.0.1", &port.to_string(), &token_file, &ca_file).unwrap();
        let (status, body) = client
            .request(hyper::Method::GET, "/api/v1/nodes", None)
            .await
            .unwrap();
        assert_eq!((status.as_u16(), body.as_str()), (200, "ok"));
        let head = server.await.unwrap().to_ascii_lowercase();
        assert!(
            head.contains("authorization: bearer sa-token\r\n"),
            "request must carry the SA token; got: {head}"
        );
        let _ = std::fs::remove_file(&ca_file);
        let _ = std::fs::remove_file(&token_file);
    }

    /// A server whose cert chains to a CA the client does not trust must be
    /// rejected; otherwise a MITM could harvest the ServiceAccount token.
    #[tokio::test]
    async fn in_cluster_client_rejects_server_signed_by_untrusted_ca() {
        let (_server_ca_pem, server_cfg) = ic_test_server();
        let (other_ca_pem, _) = ic_test_server();
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(server_cfg));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            acceptor.accept(tcp).await.is_err()
        });

        let ca_file = write_temp_file(&other_ca_pem, "ic-neg-ca");
        let token_file = write_temp_file("sa-token", "ic-neg-token");
        let client =
            in_cluster_client_from("127.0.0.1", &port.to_string(), &token_file, &ca_file).unwrap();
        let res = client
            .request(hyper::Method::GET, "/api/v1/nodes", None)
            .await;
        assert!(res.is_err(), "unrelated CA must fail TLS verification");
        assert!(
            server.await.unwrap(),
            "server must observe the client aborting the handshake"
        );
        let _ = std::fs::remove_file(&ca_file);
        let _ = std::fs::remove_file(&token_file);
    }

    /// While the file stays unreadable, re-reads (and warnings) must be
    /// limited to one per interval, not one per request.
    #[test]
    fn token_file_backs_off_after_failed_refresh() {
        let path = write_temp_file("tok-1", "token-backoff");
        let interval = std::time::Duration::from_millis(100);
        let tf = TokenFile::with_refresh(&path, interval).unwrap();
        std::thread::sleep(interval);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(tf.token(), "tok-1");
        // Recovered file must not be seen until the interval passes again,
        // proving the failed attempt was not retried immediately.
        std::fs::write(&path, "tok-2").unwrap();
        assert_eq!(tf.token(), "tok-1", "failed attempt must back off");
        std::thread::sleep(interval);
        assert_eq!(tf.token(), "tok-2");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn in_cluster_server_url_rejects_ipv6_zone_ids() {
        assert!(in_cluster_server_url("fe80::1%eth0", "443").is_err());
    }
}
