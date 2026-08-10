# Install nginx
sudo apt install -y nginx

# Create the default nginx path using nano and write the following configuration in it
sudo nano /etc/nginx/sites-available/default

    server {
        listen 80;
        server_name 3.235.232.170;

        location / {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }

# Test Nginx, Restart Nginx process, Configure to start on boot
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx