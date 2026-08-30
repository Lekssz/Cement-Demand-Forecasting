# AWS Account Setup Guide

This guide covers the initial AWS account setup required before deploying the Cement Demand Forecasting application.

## 1. Create an AWS Account

Go to the AWS website and select **Create an AWS Account**.

Provide:

- A valid email address
- An AWS account name
- A strong root-user password

Verify the email address using the verification code sent by AWS.

## 2. Enter Contact and Payment Information

Choose either a **Personal** or **Business** account depending on the intended use.

Provide the required:

- Name
- Address
- Phone number
- Payment card details

AWS requires a valid payment method even when using services that may qualify for Free Tier benefits or account credits.

## 3. Verify Your Identity

Complete the AWS identity verification process using the phone number provided during registration.

Follow the verification instructions shown by AWS.

## 4. Select a Support Plan

For development and testing, the **Basic Support Plan** is sufficient.

Paid support plans are not required for this project.

## 5. Sign In to the AWS Management Console

After account creation is complete, sign in to the AWS Management Console.

For this project, the deployment was created in:

- **Region:** Europe (London)
- **Region code:** `eu-west-2`

Keeping the resources in one region makes the deployment easier to manage.

## 6. Secure the Root Account with MFA

Multi-Factor Authentication (MFA) should be enabled for the AWS root account.

From the AWS Console:

1. Open **Security Credentials**
2. Locate **Multi-Factor Authentication (MFA)**
3. Select **Assign MFA device**
4. Configure either a passkey, authenticator application, or another supported MFA method

For the project account, the MFA device can be given a descriptive name such as:

`amdari-mig`

The root account should only be used for account-level administration.

## 7. Configure Billing and Cost Monitoring

AWS resources can generate charges while they are running, so a budget should be configured before or shortly after deployment.

Go to:

**Billing and Cost Management → Budgets → Create Budget**

For a temporary test deployment, an example configuration is:

- **Budget type:** Cost budget
- **Period:** Monthly
- **Budget name:** `mig-test-budget`
- **Budget amount:** `$5`

An alert can be configured when actual spending reaches approximately:

- **80% of the budget**
- **100% of the budget**

The notification email should be an address that is checked regularly.

A budget alert provides a warning but does not automatically stop AWS resources.

## 8. Prepare for EC2 Deployment

After the AWS account has been created and secured, the next stage is to create an EC2 instance that will host the Cement Demand Forecasting application.

The EC2 deployment process is documented separately in:

`launch-instance.md`

The server configuration and application deployment are documented in:

`deploy-to-server.md`

Nginx configuration is documented in:

`setting-up-nginx.md`