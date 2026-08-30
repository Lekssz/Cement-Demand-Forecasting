# Provision the EC2 Instance

## Via AWS Console

- Go to **EC2 → Instances → Launch Instance**
- Name the instance: `mig-cement-forecasting`
- Select **Ubuntu Server 24.04 LTS (64-bit x86)**
- Choose `t3.small` for the test deployment
- Create/select a `.pem` key pair, e.g. `mig-cement-key`
- Configure inbound rules:
  - SSH (22) → **My IP**
  - HTTP (80) → **Anywhere**
- Configure storage: **20 GiB gp3**
- Launch the instance and wait for **2/2 status checks passed**
- Copy the **Public IPv4 address**

## Connect with SSH

```bash
chmod 400 mig-cement-key.pem
ssh -i mig-cement-key.pem ubuntu@YOUR_PUBLIC_IP