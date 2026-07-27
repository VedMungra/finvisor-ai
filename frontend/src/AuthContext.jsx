import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

const AuthContext = createContext(null);

const TOKEN_KEY = 'finvisor_token';
const USER_KEY = 'finvisor_user';

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const stored = localStorage.getItem(USER_KEY);
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY) || null);
  const [loading, setLoading] = useState(!!localStorage.getItem(TOKEN_KEY));
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => { isMountedRef.current = false; };
  }, []);

  // On mount, verify the stored token is still valid
  useEffect(() => {
    const storedToken = localStorage.getItem(TOKEN_KEY);
    if (!storedToken) {
      setLoading(false);
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(`${API_URL}/auth/me`, {
          headers: { Authorization: `Bearer ${storedToken}` },
        });
        if (!response.ok) throw new Error('Invalid token');
        const data = await response.json();
        if (cancelled || !isMountedRef.current) return;
        setUser(data.user);
        setToken(storedToken);
        localStorage.setItem(USER_KEY, JSON.stringify(data.user));
      } catch {
        if (cancelled || !isMountedRef.current) return;
        // Token expired or invalid — clear auth state
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        setToken(null);
        setUser(null);
      } finally {
        if (!cancelled && isMountedRef.current) setLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, []);

  const _setAuth = useCallback((tokenValue, userValue) => {
    setToken(tokenValue);
    setUser(userValue);
    if (tokenValue) {
      localStorage.setItem(TOKEN_KEY, tokenValue);
      localStorage.setItem(USER_KEY, JSON.stringify(userValue));
    } else {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    }
  }, []);

  const login = useCallback(async (email, password) => {
    const response = await fetch(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Login failed (${response.status})`);
    }

    const data = await response.json();
    _setAuth(data.token, data.user);
    return data.user;
  }, [_setAuth]);

  const register = useCallback(async (username, email, password) => {
    const response = await fetch(`${API_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Registration failed (${response.status})`);
    }

    const data = await response.json();
    _setAuth(data.token, data.user);
    return data.user;
  }, [_setAuth]);

  const logout = useCallback(() => {
    _setAuth(null, null);
  }, [_setAuth]);

  const value = {
    user,
    token,
    isAuthenticated: !!token && !!user,
    loading,
    login,
    register,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
