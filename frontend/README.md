# CivicAccess Frontend

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![React](https://img.shields.io/badge/react-19.2.0-61dafb.svg)
![Tailwind](https://img.shields.io/badge/tailwind%20css-4.1.17-38b2ac.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

The React-based frontend for CivicAccess - a multilingual legal chatbot providing accessible information on Nigerian law.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Components](#components)
- [Styling](#styling)
- [Scripts](#scripts)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Screenshots](#screenshots)

## 🎯 Overview

CivicAccess Frontend is a modern React application that provides an intuitive interface for users to access legal information on Nigerian law. It references the 1999 Nigerian Constitution, the Nigerian Police Act, and Lagos State Tenancy Laws, available in five languages.

**Key Features:**
- Responsive design for all devices
- Multilingual support (5 languages)
- Real-time chat interface
- User authentication system
- Legal disclaimer and terms

## ✨ Features

### 🌍 Multilingual Support
- **English** - Default language
- **Hausa** - Northern Nigeria
- **Igbo** - Eastern Nigeria
- **Yoruba** - Western Nigeria
- **Pidgin English** - Cross-regional

Users can switch languages instantly without losing context.

### 💬 Chat Interface
- Real-time message input and display
- AI-powered legal assistance
- Message history within session
- Typing indicators
- Message validation

### 👤 User Authentication
- User registration/signup
- Secure login with JWT tokens
- Session management
- Password reset functionality
- User profile management

### 📱 Responsive Design
- **Mobile-first** approach
- Mobile: < 768px
- Tablet: 768px - 1024px
- Desktop: > 1024px

### 🎨 User Experience
- Clean, intuitive interface
- Easy language switching
- Legal disclaimers and terms
- Footer with project information
- Loading states and error handling

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 19.2.0 | UI Framework |
| React Router DOM | 7.10.1 | Client-side routing |
| Vite | 7.2.7 | Build tool & dev server |
| Tailwind CSS | 4.1.17 | Utility-first styling |
| Lucide React | 0.556.0 | Icon library |
| React Icons | 5.5.0 | Additional icons |
| ESLint | 9.39.1 | Code linting |

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v14 or higher)
- **npm** (comes with Node.js) or **yarn**
- **Git**
- Backend API running at `http://localhost:8000`

### Installation

#### 1. Navigate to Frontend Directory

```bash
cd Civic_Access/CivicAccess
```

#### 2. Install Dependencies

```bash
npm install
```

#### 3. Create Environment Variables

```bash
# Copy the example environment file
cp .env.example .env.local
```

Edit `.env.local`:

```env
# Backend API URL
VITE_BACKEND_URL=http://localhost:8000

# API Configuration
VITE_API_KEY=your_api_key_here

# App Configuration
VITE_APP_TITLE=CivicAccess
```

#### 4. Start Development Server

```bash
npm run dev
```

The application will open at `http://localhost:5173`

## 📁 Project Structure

```
src/
├── components/                  # Reusable components
│   ├── Footer.jsx              # Footer with disclaimers
│   ├── Background.jsx          # Background pattern/decoration
│   └── ...                     # Additional components
│
├── pages/                       # Page components (routed)
│   ├── Login.jsx               # User login page
│   ├── SignUp.jsx              # User registration page
│   ├── Chatbot.jsx             # Main chatbot interface
│   ├── LanguageSelection.jsx   # Language choice page
│   ├── ForgotPassword.jsx      # Password recovery
│   └── ...                     # Additional pages
│
├── assets/                      # Static assets
│   ├── images/                 # Image files
│   ├── icons/                  # Icon files
│   └── ...
│
├── App.jsx                      # Main app component
├── App.css                      # App-level styles
├── main.jsx                     # React entry point
├── index.css                    # Global styles
└── index.html                   # HTML template
```

## 🧩 Components

### Layout Components

#### Header
- Displays CivicAccess branding
- Language selector dropdown
- Login/Logout button
- Navigation menu
- Online status indicator

#### Footer
- Copyright information
- Team Sabilaw credit
- Hover-reveal tooltip with:
  - CivicAccess mission statement
  - Legal disclaimer
  - Terms and conditions notice
- Links to social media (optional)

#### Background
- Decorative background pattern
- Helps with visual hierarchy
- Responsive to screen size

### Page Components

#### Login Page
- Email/password input
- "Remember me" checkbox
- "Forgot password" link
- Sign up redirect
- Form validation

#### SignUp Page
- Full name, email, password fields
- Password confirmation
- Terms acceptance checkbox
- Login redirect
- Input validation

#### Chatbot Page
- Message display area
- Real-time message input
- Send button with validation
- Language selector
- Message history display

#### Language Selection
- Language options (5 languages)
- Flag/icon representation
- Easy selection interface

#### Forgot Password
- Email input field
- Reset link sending
- Confirmation message
- Return to login link

## 🎨 Styling

### Tailwind CSS Configuration

CivicAccess uses Tailwind CSS v4 for all styling.

### Design System

#### Colors
| Element | Color | Hex | Purpose |
|---------|-------|-----|---------|
| Primary | Green | #16a34a | Trust, legal authority |
| Background | White/Gray | #f9fafb | Clean readability |
| Text | Dark Gray | #111827 | Primary content |
| Secondary Text | Gray | #6b7280 | Secondary info |
| Error | Red | #dc2626 | Warnings/errors |
| Success | Green | #10b981 | Success messages |

#### Typography
- **Heading (h1-h3):** Bold, larger sizes for hierarchy
- **Body:** Regular weight for main content
- **Small:** For captions and metadata
- **Font:** System default or custom (Segoe UI, -apple-system)

#### Spacing
- Uses Tailwind's spacing scale (4px base unit)
- Consistent padding and margins
- Responsive spacing with breakpoints

### Responsive Breakpoints

```css
/* Mobile First */
/* xs: 0px - default */
/* sm: 640px - small devices */
/* md: 768px - tablets */
/* lg: 1024px - desktops */
/* xl: 1280px - large screens */
```

Example:
```jsx
<div className="p-4 md:p-8 lg:p-12">
  Responsive padding
</div>
```

## 📜 Available Scripts

### Development

```bash
# Start Vite dev server with hot reload
npm run dev
```

### Production Build

```bash
# Build optimized production bundle
npm run build
```

Output: `dist/` directory

### Preview Production Build

```bash
# Locally preview production build
npm run preview
```

### Code Linting

```bash
# Check code with ESLint
npm run lint

# Fix linting issues (auto-fix where possible)
npm run lint -- --fix
```

## 🌐 Routing

The app uses React Router v7 for client-side navigation:

```jsx
<BrowserRouter>
  <Routes>
    <Route path="/" element={<HomePage />} />
    <Route path="/login" element={<Login />} />
    <Route path="/signup" element={<SignUp />} />
    <Route path="/chatbot" element={<Chatbot />} />
    <Route path="/language-selection" element={<LanguageSelection />} />
    <Route path="/forgot-password" element={<ForgotPassword />} />
  </Routes>
</BrowserRouter>
```

## 📊 Performance Optimizations

### Code Splitting
- Lazy-loaded route components
- Reduces initial bundle size
- Faster page loads

### Caching
- Browser caching for assets
- Service Worker ready (optional)

### Image Optimization
- Optimized image assets
- Responsive image loading
- WebP format where supported

### Minification
- Production build minifies CSS/JS
- Removes unused code
- ~60-70% size reduction from dev to prod

## 🚢 Deployment

### Build for Production

```bash
npm run build
```

Creates optimized build in `dist/` directory.

### Deployment Platforms

#### Vercel (Recommended)
```bash
npm install -g vercel
vercel
```

#### Netlify
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=dist
```

#### Traditional Server
```bash
# Copy dist/ contents to your web server
# Configure server for SPA (single page app)
# Set up reverse proxy for API calls
```

### Environment Variables for Deployment

Create `.env.production`:

```env
VITE_BACKEND_URL=https://your-api-domain.com
VITE_API_KEY=your_production_key
```

## ♿ Accessibility

- Semantic HTML structure
- ARIA labels for interactive elements
- Keyboard navigation support
- Color contrast compliance (WCAG)
- Mobile-friendly touch targets

## 🧪 Testing (Future)

```bash
# Install testing dependencies
npm install --save-dev vitest @testing-library/react

# Run tests
npm run test

# Coverage report
npm run test:coverage
```

## 🌐 Browser Support

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | Latest | ✅ Fully Supported |
| Firefox | Latest | ✅ Fully Supported |
| Safari | Latest | ✅ Fully Supported |
| Edge | Latest | ✅ Fully Supported |
| Mobile Browsers | Latest | ✅ Fully Supported |

## 🐛 Troubleshooting

### Common Issues

**Port 5173 already in use**
```bash
npm run dev -- --port 5174
```

**Module not found errors**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Environment variables not loading**
```bash
# Ensure .env.local exists and Vite is restarted
# Variables must be prefixed with VITE_
```

**API connection errors**
```bash
# Check VITE_BACKEND_URL matches your backend
# Ensure backend CORS allows localhost:5173
```

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/AmazingFeature`
2. Make changes and commit: `git commit -m 'Add AmazingFeature'`
3. Push to branch: `git push origin feature/AmazingFeature`
4. Open a Pull Request
5. Do **not** push directly to main branch

## 🔒 Legal Disclaimer

CivicAccess provides legal information for educational purposes only. It does **not** constitute professional legal advice. Every translation is subject to interpretation. Users should consult with qualified legal professionals for specific legal matters.

## 📸 Screenshots

### Main Page
![Main Page](https://github.com/user-attachments/assets/56beef8a-9369-4c00-bcb4-290bc0f49ddc)

### Login Page
![Login Page](https://github.com/user-attachments/assets/914c4218-a979-4a30-9388-554e5cd64e4c)

### Sign Up Page
![Sign Up Page](https://github.com/user-attachments/assets/2ecd91d2-f893-43b1-add7-8c10b0d68096)

### Chatbot Page
![Chatbot Page](https://github.com/user-attachments/assets/a1979b9e-6637-48a6-8c4a-2a094a7e6bc8)

## 📞 Support & Contact

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact Team Sabilaw
- Check the main [README](../../README.md) for more info

---

**Last Updated:** December 2025  
**Version:** 1.0.0  
**Status:** Active Development  
**Maintained by:** Team Sabilaw


