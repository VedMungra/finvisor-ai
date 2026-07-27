import { useState, useCallback } from 'react';
import { useAuth } from './AuthContext';
import { LogIn, UserPlus, Eye, EyeOff, ArrowRight } from 'lucide-react';

function AuthPage() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState('login'); // 'login' | 'register'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = useCallback(
    async (e) => {
      e.preventDefault();
      setError('');
      setLoading(true);

      try {
        if (mode === 'login') {
          await login(email.trim(), password);
        } else {
          if (!username.trim()) {
            setError('Username is required.');
            setLoading(false);
            return;
          }
          if (password.length < 8) {
            setError('Password must be at least 8 characters.');
            setLoading(false);
            return;
          }
          await register(username.trim(), email.trim(), password);
        }
      } catch (err) {
        setError(err.message || 'Something went wrong.');
      } finally {
        setLoading(false);
      }
    },
    [mode, email, password, username, login, register]
  );

  const toggleMode = () => {
    setMode((prev) => (prev === 'login' ? 'register' : 'login'));
    setError('');
  };

  return (
    <div className="auth-page">
      {/* Ambient animated background orbs */}
      <div className="auth-bg-orb auth-bg-orb-1" />
      <div className="auth-bg-orb auth-bg-orb-2" />
      <div className="auth-bg-orb auth-bg-orb-3" />

      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-logo">📈 Finvisor AI</h1>
          <p className="auth-subtitle">
            {mode === 'login'
              ? 'Sign in to your account'
              : 'Create a new account'}
          </p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {mode === 'register' && (
            <div className="auth-field">
              <label className="auth-label" htmlFor="auth-username">
                Username
              </label>
              <input
                id="auth-username"
                className="auth-input"
                type="text"
                placeholder="Your display name"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                maxLength={50}
              />
            </div>
          )}

          <div className="auth-field">
            <label className="auth-label" htmlFor="auth-email">
              Email
            </label>
            <input
              id="auth-email"
              className="auth-input"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              required
            />
          </div>

          <div className="auth-field">
            <label className="auth-label" htmlFor="auth-password">
              Password
            </label>
            <div className="auth-password-wrapper">
              <input
                id="auth-password"
                className="auth-input"
                type={showPassword ? 'text' : 'password'}
                placeholder={mode === 'register' ? 'Minimum 8 characters' : 'Your password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                required
              />
              <button
                type="button"
                className="auth-eye-btn"
                onClick={() => setShowPassword((v) => !v)}
                tabIndex={-1}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>

          {error && (
            <div className="auth-error" role="alert">
              {error}
            </div>
          )}

          <button type="submit" className="auth-submit-btn" disabled={loading}>
            {loading ? (
              <span className="loader" style={{ width: '18px', height: '18px', borderWidth: '2px' }} />
            ) : mode === 'login' ? (
              <>
                <LogIn size={18} />
                Sign In
              </>
            ) : (
              <>
                <UserPlus size={18} />
                Create Account
              </>
            )}
          </button>
        </form>

        <div className="auth-toggle">
          <span className="auth-toggle-text">
            {mode === 'login' ? "Don't have an account?" : 'Already have an account?'}
          </span>
          <button type="button" className="auth-toggle-btn" onClick={toggleMode}>
            {mode === 'login' ? 'Sign Up' : 'Sign In'}
            <ArrowRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default AuthPage;
