use super::*;
use base64::Engine;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_rustls::TlsAcceptor;

const FAST: Duration = Duration::from_millis(50);

/// A complete PKI generation: one CA that signs the server leaf (for
/// 127.0.0.1) and the client leaf.
struct World {
    ca_pem: String,
    server_cert: CertificateDer<'static>,
    server_key: PrivateKeyDer<'static>,
    client_cert_pem: String,
    client_key_pem: String,
}

fn world(name: &str) -> World {
    use rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, Issuer, KeyPair, SanType};
    let ca_key = KeyPair::generate().unwrap();
    let mut ca_params = CertificateParams::default();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params
        .distinguished_name
        .push(DnType::CommonName, format!("{name}-ca"));
    let ca_cert = ca_params.self_signed(&ca_key).unwrap();
    let issuer = Issuer::new(ca_params, ca_key);

    let server_key = KeyPair::generate().unwrap();
    let mut sp = CertificateParams::default();
    sp.subject_alt_names = vec![SanType::IpAddress("127.0.0.1".parse().unwrap())];
    let server_cert = sp.signed_by(&server_key, &issuer).unwrap();

    let client_key = KeyPair::generate().unwrap();
    let mut cp = CertificateParams::default();
    cp.distinguished_name
        .push(DnType::CommonName, format!("{name}-client"));
    let client_cert = cp.signed_by(&client_key, &issuer).unwrap();

    World {
        ca_pem: ca_cert.pem(),
        server_cert: server_cert.der().clone(),
        server_key: PrivateKeyDer::Pkcs8(server_key.serialize_der().into()),
        client_cert_pem: client_cert.pem(),
        client_key_pem: client_key.serialize_pem(),
    }
}

impl World {
    /// Server config that presents this world's server cert and demands a
    /// client cert signed by this world's CA.
    fn acceptor(&self) -> TlsAcceptor {
        let provider = Arc::new(rustls_post_quantum::provider());
        let mut roots = rustls::RootCertStore::empty();
        roots
            .add(
                CertificateDer::pem_slice_iter(self.ca_pem.as_bytes())
                    .next()
                    .unwrap()
                    .unwrap(),
            )
            .unwrap();
        let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
            Arc::new(roots),
            provider.clone(),
        )
        .build()
        .unwrap();
        let cfg = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_client_cert_verifier(verifier)
            .with_single_cert(vec![self.server_cert.clone()], self.server_key.clone_key())
            .unwrap();
        TlsAcceptor::from(Arc::new(cfg))
    }

    fn write_files(&self, dir: &Path) {
        replace(dir, "ca.pem", &self.ca_pem);
        replace(dir, "client.pem", &self.client_cert_pem);
        replace(dir, "client.key", &self.client_key_pem);
    }
}

/// Atomic replace, as cert-manager / the kubelet do.
fn replace(dir: &Path, name: &str, content: &str) {
    let tmp = dir.join(format!(".{name}.tmp"));
    std::fs::write(&tmp, content).unwrap();
    std::fs::rename(&tmp, dir.join(name)).unwrap();
}

struct Server {
    port: u16,
    acceptor: Arc<Mutex<TlsAcceptor>>,
    watch_conns: Arc<AtomicUsize>,
}

impl Server {
    /// Answers `ok` to plain requests; a path containing `watch` gets one
    /// event and is then held open.
    async fn start(w: &World) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let acceptor = Arc::new(Mutex::new(w.acceptor()));
        let watch_conns = Arc::new(AtomicUsize::new(0));
        let (acc, wc) = (acceptor.clone(), watch_conns.clone());
        tokio::spawn(async move {
            loop {
                let (tcp, _) = listener.accept().await.unwrap();
                let acceptor = acc.lock().unwrap().clone();
                let wc = wc.clone();
                tokio::spawn(async move {
                    let Ok(mut tls) = acceptor.accept(tcp).await else {
                        return;
                    };
                    let mut buf = vec![0u8; 4096];
                    let mut n = 0;
                    while !buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                        match tls.read(&mut buf[n..]).await {
                            Ok(0) | Err(_) => return,
                            Ok(r) => n += r,
                        }
                    }
                    if String::from_utf8_lossy(&buf[..n]).contains("watch") {
                        wc.fetch_add(1, Ordering::SeqCst);
                        let _ = tls
                            .write_all(
                                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n\
                                  11\r\n{\"type\":\"ADDED\"}\n\r\n",
                            )
                            .await;
                        let _ = tls.flush().await;
                        tokio::time::sleep(Duration::from_secs(60)).await;
                    } else {
                        let _ = tls
                            .write_all(
                                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
                            )
                            .await;
                        let _ = tls.shutdown().await;
                    }
                });
            }
        });
        Self {
            port,
            acceptor,
            watch_conns,
        }
    }

    fn rotate_to(&self, w: &World) {
        *self.acceptor.lock().unwrap() = w.acceptor();
    }
}

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "politeia-reload-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn file_kubeconfig(dir: &Path, port: u16) -> String {
    let kc = dir.join("kubeconfig");
    std::fs::write(
        &kc,
        format!(
            "clusters:\n- cluster:\n    server: https://127.0.0.1:{port}\n    \
             certificate-authority: ca.pem\n  name: t\nusers:\n- name: u\n  user:\n    \
             client-certificate: ./client.pem\n    client-key: client.key\n"
        ),
    )
    .unwrap();
    kc.to_str().unwrap().to_owned()
}

fn data_kubeconfig(dir: &Path, port: u16, w: &World) -> String {
    let b64 = base64::engine::general_purpose::STANDARD;
    let kc = dir.join("kubeconfig-data");
    std::fs::write(
        &kc,
        format!(
            "clusters:\n- cluster:\n    server: https://127.0.0.1:{port}\n    \
             certificate-authority-data: {}\n  name: t\nusers:\n- name: u\n  user:\n    \
             client-certificate-data: {}\n    client-key-data: {}\n",
            b64.encode(&w.ca_pem),
            b64.encode(&w.client_cert_pem),
            b64.encode(&w.client_key_pem),
        ),
    )
    .unwrap();
    kc.to_str().unwrap().to_owned()
}

async fn get_ok(c: &HyperApiClient) -> anyhow::Result<()> {
    let (status, body) = c.request(Method::GET, "/healthz", None).await?;
    anyhow::ensure!(
        status.is_success() && body == "ok",
        "unexpected {status} {body}"
    );
    Ok(())
}

async fn eventually_ok(c: &HyperApiClient) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match get_ok(c).await {
            Ok(()) => return,
            Err(e) if std::time::Instant::now() > deadline => {
                panic!("client never picked up the new material: {e:#}")
            }
            Err(_) => tokio::time::sleep(FAST).await,
        }
    }
}

/// The scheduler must keep scheduling across a cert/CA rotation without a
/// restart: new requests use the rotated material and a watch opened under
/// the old identity is dropped so its reconnect uses the new one.
#[tokio::test]
async fn rotation_on_disk_reaches_new_requests_and_reconnects_open_watch() {
    let (a, b) = (world("a"), world("b"));
    let dir = tmp_dir("rotate");
    a.write_files(&dir);
    let server = Server::start(&a).await;
    let client = Arc::new(
        kubeconfig_client_with_reload(&file_kubeconfig(&dir, server.port), Some(FAST)).unwrap(),
    );
    get_ok(&client).await.expect("works before rotation");

    let events = Arc::new(AtomicUsize::new(0));
    let (c2, ev2) = (client.clone(), events.clone());
    let watch = tokio::spawn(async move {
        c2.watch_stream("/x?watch=true", move |_| {
            ev2.fetch_add(1, Ordering::SeqCst);
        })
        .await
    });
    tokio::time::sleep(FAST * 6).await;
    assert_eq!(events.load(Ordering::SeqCst), 1, "watch is live");
    assert!(!watch.is_finished(), "watch must stay open until rotation");

    server.rotate_to(&b);
    b.write_files(&dir);

    eventually_ok(&client).await;
    let r = tokio::time::timeout(Duration::from_secs(5), watch)
        .await
        .expect("open watch must be ended after the swap")
        .unwrap();
    assert!(r.is_ok(), "forced reconnect is a clean end: {r:?}");
    assert_eq!(server.watch_conns.load(Ordering::SeqCst), 1);

    let (c3, ev3) = (client.clone(), events.clone());
    let again = tokio::spawn(async move {
        c3.watch_stream("/x?watch=true", move |_| {
            ev3.fetch_add(1, Ordering::SeqCst);
        })
        .await
    });
    tokio::time::sleep(FAST * 6).await;
    assert_eq!(
        server.watch_conns.load(Ordering::SeqCst),
        2,
        "reconnected watch must handshake with the new identity"
    );
    assert!(!again.is_finished());
}

/// Control: inline `-data` kubeconfigs are fixed at startup, so after a
/// rotation the old identity is rejected. Guards the rotation test above
/// against passing for the wrong reason.
#[tokio::test]
async fn data_mode_client_cannot_follow_rotation() {
    let (a, b) = (world("a"), world("b"));
    let dir = tmp_dir("data");
    let server = Server::start(&a).await;
    let client =
        kubeconfig_client_with_reload(&data_kubeconfig(&dir, server.port, &a), Some(FAST)).unwrap();
    get_ok(&client).await.expect("works before rotation");
    server.rotate_to(&b);
    tokio::time::sleep(FAST * 4).await;
    assert!(get_ok(&client).await.is_err());
    assert_eq!(client.connector.generation(), 0);
}

/// Control: with reload disabled, file-path mode must not follow rotation
/// either; proves the swap is what makes the rotation test pass.
#[tokio::test]
async fn file_mode_with_reload_disabled_cannot_follow_rotation() {
    let (a, b) = (world("a"), world("b"));
    let dir = tmp_dir("noreload");
    a.write_files(&dir);
    let server = Server::start(&a).await;
    let client = kubeconfig_client_with_reload(&file_kubeconfig(&dir, server.port), None).unwrap();
    get_ok(&client).await.unwrap();
    server.rotate_to(&b);
    b.write_files(&dir);
    tokio::time::sleep(FAST * 4).await;
    assert!(get_ok(&client).await.is_err());
}

/// A half-written or broken replacement must never take the scheduler off the
/// working identity; a later good replacement must still be picked up.
#[tokio::test]
async fn corrupt_or_partial_replacement_keeps_working_connector() {
    let (a, b) = (world("a"), world("b"));
    let dir = tmp_dir("corrupt");
    a.write_files(&dir);
    let server = Server::start(&a).await;
    let client =
        kubeconfig_client_with_reload(&file_kubeconfig(&dir, server.port), Some(Duration::ZERO))
            .unwrap();
    get_ok(&client).await.unwrap();

    let half_cert = &a.client_cert_pem[..a.client_cert_pem.len() / 2];
    let cases: [(&str, &str, &str); 6] = [
        ("truncated cert", "client.pem", half_cert),
        ("cert/key mismatch", "client.key", &b.client_key_pem),
        ("garbage CA", "ca.pem", "not a pem"),
        ("empty CA", "ca.pem", ""),
        (
            "truncated key",
            "client.key",
            "-----BEGIN PRIVATE KEY-----\nAAAA\n",
        ),
        ("empty cert", "client.pem", ""),
    ];
    for (what, file, content) in cases {
        a.write_files(&dir);
        replace(&dir, file, content);
        get_ok(&client)
            .await
            .unwrap_or_else(|e| panic!("{what}: working connector lost: {e:#}"));
        assert_eq!(client.connector.generation(), 0, "{what}: must not swap");
    }

    std::fs::remove_file(dir.join("client.key")).unwrap();
    get_ok(&client)
        .await
        .expect("missing key file keeps connector");
    assert_eq!(client.connector.generation(), 0);

    server.rotate_to(&b);
    b.write_files(&dir);
    eventually_ok(&client).await;
    assert!(client.connector.generation() > 0);
}

/// kubeconfigs written by tooling use paths relative to the kubeconfig, and
/// the process cwd is unrelated to it.
#[tokio::test]
async fn relative_paths_resolve_against_kubeconfig_dir_not_cwd() {
    let a = world("a");
    let dir = tmp_dir("relative");
    let certs = dir.join("certs");
    std::fs::create_dir_all(&certs).unwrap();
    a.write_files(&certs);
    let server = Server::start(&a).await;
    let kc = dir.join("kubeconfig");
    std::fs::write(
        &kc,
        format!(
            "clusters:\n- cluster:\n    server: https://127.0.0.1:{}\n    \
             certificate-authority: certs/ca.pem\n  name: t\nusers:\n- name: u\n  user:\n    \
             client-certificate: \"certs/client.pem\"\n    client-key: {}\n",
            server.port,
            certs.join("client.key").display()
        ),
    )
    .unwrap();
    let client = kubeconfig_client(kc.to_str().unwrap()).expect("relative + absolute paths");
    get_ok(&client).await.unwrap();
}

/// Harvest-now-decrypt-later protection: every config this crate builds for
/// reload must offer the hybrid X25519MLKEM768 group.
#[test]
fn reloadable_client_config_offers_pq_key_exchange() {
    let a = world("a");
    let cfg = build_client_config(
        a.ca_pem.as_bytes(),
        a.client_cert_pem.as_bytes(),
        a.client_key_pem.as_bytes(),
    )
    .unwrap();
    assert!(
        cfg.crypto_provider()
            .kx_groups
            .iter()
            .any(|g| g.name() == rustls::NamedGroup::X25519MLKEM768),
        "X25519MLKEM768 must be offered"
    );
}
