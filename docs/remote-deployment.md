# Remote deployment

If you want claude-echoes accessible from multiple machines (laptop, desktop, WSL), run the server on a VPS and point each machine's hook at the remote URL.

**Warning:** your verbatim Claude prompts contain a lot — code, credentials you pasted in, client data, private conversations. Treat the database like you would a password vault. Always use TLS + auth when exposing it beyond localhost.

## 1. Run the stack on your VPS

```bash
git clone https://github.com/m4cd4r4/claude-echoes
cd claude-echoes

# Change the bind from 127.0.0.1 to 0.0.0.0 in docker-compose.yml
# OR leave it local and expose only via nginx (recommended)

docker compose up -d
docker exec echoes-ollama ollama pull nomic-embed-text
docker exec -i echoes-postgres psql -U echoes -d echoes < sql/001_init.sql
```

## 2. Put nginx in front with TLS + bearer auth

Example nginx block (assumes you already have LetsEncrypt certs):

```nginx
server {
    listen 443 ssl http2;
    server_name echoes.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/echoes.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/echoes.yourdomain.com/privkey.pem;

    location / {
        # Require Bearer token
        if ($http_authorization != "Bearer YOUR_LONG_RANDOM_TOKEN_HERE") {
            return 401;
        }
        proxy_pass http://127.0.0.1:8088;
        proxy_set_header Host $host;
    }
}
```

Generate a strong token:

```bash
openssl rand -base64 48
```

## 3. Point the hook at the remote server

On each client machine, set env vars before launching Claude Code:

```bash
export ECHOES_URL="https://echoes.yourdomain.com"
export ECHOES_TOKEN="YOUR_LONG_RANDOM_TOKEN_HERE"
```

Or add them to your shell rc file so they're always set.

## 4. Verify

```bash
curl -s -H "Authorization: Bearer $ECHOES_TOKEN" \
  "https://echoes.yourdomain.com/health"
```

Should return `{"ok":true,"db":"up","model":"nomic-embed-text"}`.
