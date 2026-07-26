import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import ErrorBoundary from './ErrorBoundary.jsx'

const container = document.getElementById('root')

if (!container) {
  // Fail loudly in the page instead of throwing into a blank screen if index.html changes.
  console.error('Finvisor: #root element not found — the app cannot mount.')
} else {
  createRoot(container).render(
    <StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </StrictMode>,
  )
}
