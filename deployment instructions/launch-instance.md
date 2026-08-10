# Provision the EC2 instance
## Via AWS Console:

- Go to EC2 → Instances → Launch Instance
- Name your instance (e.g. api-server)
- Choose an AMI — Ubuntu 22.04 LTS or Amazon Linux 2023 are good defaults
- Choose instance type — t3.micro (free-tier eligible) or t3.small for more headroom
- Create/select a key pair (.pem file) — download it, you need it for SSH
- Configure Network settings:
    Allow SSH (port 22) from "My IP" (not 0.0.0.0/0, for security)
    Allow HTTP (80) from anywhere

- Configure storage - leave at default
- Click Launch Instance