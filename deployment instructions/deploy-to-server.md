# Login to your server using SSH
`ssh -i your-key.pem ubuntu@your-public-ip`

# Set up the Server Environment

### Update packages
`sudo apt update && sudo apt upgrade -y`

### Install prerequisites
`sudo apt install -y ca-certificates curl gnupg`

### Add Docker's official GPG key
`sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg`

### Add the Docker repository
`echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`

### Install Docker and Docker Compose
`sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin`

### Let ubuntu user run docker without sudo (optional - you can go ahead with including sudo)
`sudo usermod -aG docker $USER
newgrp docker   # or log out/in to apply the group change`

### Verify
`docker --version
docker compose version

# Full Deployment Steps

- Step 1: Clone your Project from Github.
- Step 2: cd into project path.
- Step 3: Run docker build (sudo docker build -t ride-wise-project .)
- Step 4: Start up Docker Container (sudo docker run -d -p 8000:8000 ride-wise-project)
- Step 5: Set up nginx - refer to setting-up-nginx.md for the steps

### Investigate Logs incase of any error
docker logs ride-wise