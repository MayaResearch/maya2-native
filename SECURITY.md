# Security and misuse reports

Please do not report security problems or misuse in a public issue.

- **Security vulnerabilities in the engine or server** (for example a request or an uploaded reference clip that crashes the server or exhausts its resources): use GitHub's private vulnerability reporting (**Security → Report a vulnerability** on this repository), or e-mail TBD.
- **Misuse of the model** (impersonation, cloning without consent, fraud): e-mail TBD with as much detail as you can share.

## Supported versions

| version | supported |
|---|---|
| 1.0 | yes |

## Deployment notes

- `maya3.server` has no authentication. Put it behind your own gateway before exposing it outside a trusted network.
- `POST /v1/voices` accepts reference audio from the caller. Restrict it if untrusted users can reach the server, and only register voices you have consent to use.
