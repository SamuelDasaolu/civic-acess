# CivicAccess Backend API

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/framework-FastAPI-green.svg)

The backend API for CivicAccess - a multilingual legal chatbot providing accessible information on Nigerian law.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [Authentication](#authentication)
- [Development](#development)
- [Deployment](#deployment)

## 🎯 Overview

The CivicAccess backend is built with **FastAPI**, a modern, fast web framework for building APIs with Python. It provides:

- **RESTful API** endpoints for chat, authentication, and legal information retrieval
- **User Authentication** with JWT tokens
- **Database Management** using SQLAlchemy ORM with LibSQL
- **CORS Support** for cross-origin requests from the frontend
- **Legal Information** stored and retrieved efficiently

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (venv)
- LibSQL database connection (local or remote)

## 🚀 Installation

### 1. Navigate to Backend Directory

```bash
cd Civic_Access/backend
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the backend directory:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Database Configuration
DATABASE_URL=file:local.db

# JWT Configuration
SECRET_KEY=your-super-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000

# API Configuration
API_TITLE=CivicAccess Backend
API_VERSION=1.0.0
DEBUG=True
```

### 5. Initialize Database

```bash
python
>>> from database import init_db
>>> init_db()
>>> exit()
```

Or automatically on first run (handled by startup event).

### 6. Run the Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

**API Documentation:**
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## ⚙️ Configuration

### Dependencies (`requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.123.9 | Web framework |
| uvicorn | 0.38.0 | ASGI server |
| sqlalchemy | 2.0.44 | ORM |
| python-dotenv | 1.0.1 | Environment variables |
| bcrypt | 5.0.0 | Password hashing |
| PyJWT | 2.10.1 | JWT tokens |
| pydantic | 2.12.4 | Data validation |
| pydantic-settings | 2.5.0 | Settings management |
| python-multipart | 0.0.7 | Form data parsing |
| email-validator | 2.3.0 | Email validation |

### File Structure

```
backend/
├── app.py                    # Main FastAPI application
├── database.py               # Database connection & initialization
├── models.py                 # SQLAlchemy models (User, etc.)
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── local.db                  # SQLite database (created on init)
├── routes/
│   ├── __init__.py
│   ├── auth.py              # Authentication endpoints
│   └── ...                  # Additional route modules
├── myenv/                    # Virtual environment
└── README.md                 # This file
```

## 🔌 API Endpoints

### Authentication Routes

#### Register User
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword",
  "full_name": "User Name"
}

Response: 201 Created
{
  "message": "User created successfully",
  "user_id": 1
}
```

#### Login
```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=securepassword

Response: 200 OK
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "full_name": "User Name"
  }
}
```

#### Logout
```http
POST /auth/logout
Authorization: Bearer {token}

Response: 200 OK
{
  "message": "Logged out successfully"
}
```

### General Routes

#### Health Check
```http
GET /

Response: 200 OK
{
  "message": "Welcome to CivicAccess API"
}
```

## 💾 Database Schema

### Users Table

```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email VARCHAR(255) UNIQUE NOT NULL,
  hashed_password VARCHAR(255) NOT NULL,
  full_name VARCHAR(255),
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### User Model (SQLAlchemy)

```python
class User(Base):
    __tablename__ = "users"
    
    id: int = Column(Integer, primary_key=True)
    email: str = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password: str = Column(String(255), nullable=False)
    full_name: Optional[str] = Column(String(255))
    is_active: bool = Column(Boolean, default=True)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

## 🔐 Authentication

CivicAccess uses **JWT (JSON Web Tokens)** for authentication.

### How It Works

1. **Registration:** User provides email, password, and name
2. **Password Hashing:** Password is hashed using bcrypt before storage
3. **Login:** User provides credentials, receives JWT token
4. **Authorization:** Token included in `Authorization: Bearer {token}` header
5. **Validation:** Server validates token on each protected request

### Protected Routes

Protected endpoints require a valid JWT token:

```python
@app.get("/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    return {"user_id": current_user.id, "email": current_user.email}
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.
```

### Code Style & Linting

```bash
# Format code with Black
black .

# Check with flake8
flake8 .
```

### Project Development

The backend is actively under development. Key areas include:

- Legal information retrieval and processing
- LLM integration for intelligent responses
- Multilingual support for responses
- Enhanced user profile management
- Chat history and analytics
- Admin dashboard endpoints

## 🚢 Deployment

### Production Checklist

- [ ] Set `DEBUG=False` in `.env`
- [ ] Update `SECRET_KEY` with a strong random string
- [ ] Configure `ALLOWED_ORIGINS` for your domain
- [ ] Set up a production database (PostgreSQL recommended)
- [ ] Use a production ASGI server (Gunicorn + Uvicorn)
- [ ] Set up SSL/TLS certificates
- [ ] Configure environment variables securely

### Deploy with Gunicorn

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn (4 workers, port 8000)
gunicorn app:app -w 4 -b 0.0.0.0:8000 --timeout 120
```

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t civicaccess-backend .
docker run -p 8000:8000 civicaccess-backend
```

## 📝 Additional Notes

### CORS Configuration

The backend allows requests from:
- `http://localhost:5173` (React dev server)
- `http://127.0.0.1:8000` (Local testing)
- Configured `VITE_BACKEND_URL` from environment

Modify `origins` list in `app.py` for additional domains.

### Database

Currently uses SQLite (`local.db`) for development. For production, consider:
- **PostgreSQL** - More robust, better for scaling
- **MySQL** - Alternative relational database
- **LibSQL** - Cloud-based SQLite compatible database

## 🐛 Troubleshooting

### Common Issues

**Issue: ModuleNotFoundError**
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

**Issue: Database locked**
```bash
# Solution: Delete local.db and reinitialize
rm local.db
python -c "from database import init_db; init_db()"
```

**Issue: Port 8000 already in use**
```bash
# Solution: Run on different port
uvicorn app:app --port 8001 --reload
```

## 📞 Support

For issues or questions:
- Check the [main README](../../README.md)
- Open an issue on GitHub
- Contact Team Sabilaw

---

**Last Updated:** December 2025  
**Version:** 1.0.0  
**Status:** Active Development  
**Maintained by:** Team Sabilaw
