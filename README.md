# Heart Disease Prediction System

A Django-based web application for heart disease prediction using machine learning, ECG image analysis, AI-powered calling agent for hospital appointment booking, and medical chat assistance.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Clone the Repository](#clone-the-repository)
3. [Install Python Dependencies](#install-python-dependencies)
4. [Get API Keys](#get-api-keys)
   - [Twilio (Calling, SMS, WhatsApp)](#twilio-calling-sms-whatsapp)
   - [Claude API (AI Conversations)](#claude-api-ai-conversations)
   - [Sarvam AI (Voice / Speech-to-Text)](#sarvam-ai-voice--speech-to-text)
   - [Mapbox (Doctor Finder Map)](#mapbox-doctor-finder-map)
5. [Configure Environment Variables](#configure-environment-variables)
6. [Run Database Migrations](#run-database-migrations)
7. [Install and Start ngrok](#install-and-start-ngrok)
8. [Update BASE_URL in .env](#update-base_url-in-env)
9. [Run the Django Server](#run-the-django-server)
10. [Default Login Credentials](#default-login-credentials)

---

## Prerequisites

Make sure you have the following installed:

- **Python 3.11+** — [https://www.python.org/downloads/](https://www.python.org/downloads/)
- **pip** — comes with Python
- **ngrok** — [https://ngrok.com/download](https://ngrok.com/download)
- **Git** — [https://git-scm.com/](https://git-scm.com/)

---

## Clone the Repository

```bash
git clone https://github.com/himanshusdeshmukh2106/oop-cp.git
cd oop-cp
```

---

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

Then install the additional packages required by the Django app:

```bash
pip install twilio python-dotenv google-generativeai reportlab requests
```

---

## Get API Keys

### Twilio (Calling, SMS, WhatsApp)

The AI calling agent uses Twilio to make real phone calls to hospitals and send SMS/WhatsApp notifications.

1. Go to [https://console.twilio.com/](https://console.twilio.com/) and create a free account.
2. After signing in, on the **Console Dashboard** you will find:
   - **Account SID** — starts with `AC`, 34 characters long
   - **Auth Token** — 32 characters (click the eye icon to reveal it)
3. Get a Twilio phone number:
   - Go to **Phone Numbers > Manage > Buy a number**
   - Buy a US number with **Voice** and **SMS** capabilities (free trial gives $15 credit)
   - Note the number in E.164 format, e.g. `+19063656394`
4. For **WhatsApp** (optional):
   - Go to **Messaging > Try it out > Send a WhatsApp message**
   - Follow the sandbox setup instructions
   - Send `join <your-sandbox-code>` to `+14155238886` on WhatsApp from your phone
   - Note your **WhatsApp Sandbox Number** (usually `+14155238886`)

### Claude API (AI Conversations)

The AI calling agent uses a Claude-compatible API to generate intelligent responses during phone calls.

1. Go to [https://api.quatarly.cloud/](https://api.quatarly.cloud/) and register for an account.
2. Generate an API key from your dashboard.
3. Copy the key — it will be used as `CLAUDE_API_KEY`.

> Alternatively, if you have an Anthropic account, get your key from [https://console.anthropic.com/](https://console.anthropic.com/).

### Sarvam AI (Voice / Speech-to-Text)

Used for multilingual speech-to-text, especially Indian languages.

1. Go to [https://dashboard.sarvam.ai/](https://dashboard.sarvam.ai/) and sign up.
2. From your dashboard, create an API key.
3. Copy the key — it will be used as `SARVAM_API_KEY`.

### Mapbox (Doctor Finder Map)

Used to display an interactive map for finding nearby hospitals and doctors.

1. Go to [https://account.mapbox.com/](https://account.mapbox.com/) and create a free account.
2. Go to **Access Tokens** from the dashboard.
3. Copy your **Default public token** (starts with `pk.eyJ1...`).
4. This will be used as `MAPBOX_API_KEY`.

---

## Configure Environment Variables

Create a `.env` file in the **root of the repository** (same level as this README):

```env
# Django
SECRET_KEY='your-random-secret-key-here'
DEBUG=True

# Claude AI (for AI calling agent conversation)
CLAUDE_API_KEY='your-claude-api-key-here'

# Sarvam AI - Multilingual Speech-to-Text
# Get from: https://dashboard.sarvam.ai/
SARVAM_API_KEY=your-sarvam-api-key-here

# Base URL for Twilio callbacks — replace after starting ngrok (Step 7)
BASE_URL='https://your-ngrok-url.ngrok-free.app'

# Mapbox — for doctor finder map
# Get from: https://account.mapbox.com/access-tokens/
MAPBOX_API_KEY='pk.eyJ1...your-mapbox-token'

# Twilio Credentials
# Get from: https://console.twilio.com/
TWILIO_ACCOUNT_SID='ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
TWILIO_AUTH_TOKEN='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
TWILIO_PHONE_NUMBER='+1xxxxxxxxxx'

# Your own phone number for testing (all calls go here during development)
TEST_PHONE_NUMBER='+91xxxxxxxxxx'

# WhatsApp (optional)
WHATSAPP_SANDBOX_NUMBER='+14155238886'
TEST_SMS_PHONE_NUMBER='+91xxxxxxxxxx'
```

> **Important:** Never commit the `.env` file to git. It is already listed in `.gitignore`.

---

## Run Database Migrations

Navigate into the Django project directory and run migrations:

```bash
cd Heart-Disease-Prediction-System
python manage.py migrate
```

Create a superuser (admin account):

```bash
python manage.py createsuperuser
```

Follow the prompts to set a username, email, and password.

---

## Install and Start ngrok

ngrok creates a public HTTPS tunnel to your local Django server. This is required for Twilio to send webhook callbacks to your machine during AI calls.

### Step 1 — Download ngrok

Go to [https://ngrok.com/download](https://ngrok.com/download) and download the installer for your OS.

On Windows:
- Download the ZIP, extract `ngrok.exe`, and place it somewhere on your PATH (e.g. `C:\ngrok\ngrok.exe`).

### Step 2 — Create a free ngrok account and get your auth token

1. Sign up at [https://dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup)
2. After login, go to [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
3. Copy your auth token.

### Step 3 — Authenticate ngrok

```bash
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

### Step 4 — Start ngrok tunnel

In a **separate terminal window**, run:

```bash
ngrok http 8000
```

You will see output like:

```
Forwarding    https://6591-223-228-139-87.ngrok-free.app -> http://localhost:8000
```

Copy the `https://...ngrok-free.app` URL — this is your public URL.

---

## Update BASE_URL in .env

Open your `.env` file and update `BASE_URL` with the ngrok URL you just copied:

```env
BASE_URL='https://6591-223-228-139-87.ngrok-free.app'
```

Save the file. The Django server will pick this up automatically (you may need to restart it if it is already running).

---

## Run the Django Server

Make sure you are inside the `Heart-Disease-Prediction-System` directory:

```bash
cd Heart-Disease-Prediction-System
python manage.py runserver
```

The application will be available at:

- Local: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Public (via ngrok): `https://your-ngrok-url.ngrok-free.app`

---

## Default Login Credentials

After running migrations and creating a superuser, use the credentials you set during `createsuperuser` to log in at:

```
http://127.0.0.1:8000/admin
```

For the main application login page:

```
http://127.0.0.1:8000/login
```

Register a new patient or doctor account, or use the admin panel to create users directly.

---

## Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install twilio python-dotenv google-generativeai reportlab requests

# 2. Set up .env file with all API keys (see above)

# 3. Run migrations
cd Heart-Disease-Prediction-System
python manage.py migrate

# 4. In a separate terminal, start ngrok
ngrok http 8000

# 5. Copy the ngrok URL and update BASE_URL in .env

# 6. Start Django server
python manage.py runserver
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
