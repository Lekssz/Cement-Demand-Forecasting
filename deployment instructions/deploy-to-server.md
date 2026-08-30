# Deploy to Server

## 1. Connect to EC2

```bash
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP
```

## 2. Update the Server

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg
```

## 3. Install Docker

```bash
sudo install -m 0755 -d /etc/apt/keyrings

sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc

sudo chmod a+r /etc/apt/keyrings/docker.asc
```

Add Docker repository:

```bash
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
```

Install Docker:

```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Allow Docker without sudo:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 4. Clone and Run the Project

```bash
git clone https://github.com/Lekssz/Cement-Demand-Forecasting.git
cd Cement-Demand-Forecasting/src
docker compose build
docker compose up -d
```

Check services:

```bash
docker compose ps -a
```

## 5. Verify

```bash
curl http://localhost:8000/health
curl -I http://localhost:8050
```

If successful, continue with `setting-up-nginx.md`.

## 6. Check Logs if Needed

```bash
docker compose logs
```