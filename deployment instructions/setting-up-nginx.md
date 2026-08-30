# Install Nginx

```bash
sudo apt install -y nginx
```

# Configure Nginx

Open the default configuration:

```bash
sudo nano /etc/nginx/sites-available/default
```

Replace the existing content with:

```nginx
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8050;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Port `8050` is used because the Dash dashboard is the public application.

# Test and Restart Nginx

```bash
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

Test the deployment:

```bash
curl -I http://localhost
```

Then open the EC2 Public IPv4 address in a browser:

```text
http://YOUR_PUBLIC_IP
```