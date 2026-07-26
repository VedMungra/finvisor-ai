import { Component } from 'react';

/**
 * Catches render / lifecycle errors anywhere below it. React unmounts the entire tree when a
 * render throws, so without this a single malformed message would leave the user staring at a
 * blank white page with the reason only visible in the devtools console.
 */
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
    this.handleReset = this.handleReset.bind(this);
    this.handleReload = this.handleReload.bind(this);
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('Finvisor UI render error:', error, info?.componentStack);
  }

  handleReset() {
    this.setState({ error: null });
  }

  handleReload() {
    window.location.reload();
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="app-container">
        <div className="glass-panel error-boundary">
          <h2>Something went wrong</h2>
          <p>
            The interface hit an unexpected error and stopped rendering. Your uploaded documents and
            data are unaffected.
          </p>
          <pre className="error-boundary-detail">{String(error?.message || error)}</pre>
          <div className="error-boundary-actions">
            <button type="button" className="danger-btn" onClick={this.handleReset}>
              Try again
            </button>
            <button type="button" className="danger-btn" onClick={this.handleReload}>
              Reload the app
            </button>
          </div>
        </div>
      </div>
    );
  }
}

export default ErrorBoundary;
