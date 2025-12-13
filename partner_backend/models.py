from sqlalchemy import Column, Integer, String, DateTime, Boolean
from datetime import datetime
from database import Base
from sqlalchemy.ext.declarative import declarative_base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
    

# class PasswordResetToken(Base):
#     __tablename__ = "password_reset_tokens"
    
#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String, index=True, nullable=False)
#     token = Column(String, unique=True, index=True, nullable=False)
#     created_at = Column(DateTime, default=datetime.timezone.utcnow)
#     expires_at = Column(DateTime, nullable=False)
#     used = Column(Boolean, default=False)