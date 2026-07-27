import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  UploadCloud,
  Send,
  Database,
  Globe,
  Trash2,
  ThumbsUp,
  ThumbsDown,
  CircleStop,
  SquarePen,
  ImageOff,
  RefreshCw,
  LogOut,
  User,
} from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';
import { useAuth } from './AuthContext';
import AuthPage from './AuthPage';
import PortfolioPanel from './PortfolioPanel';
import './index.css';

// VITE_API_URL is optional (fall back to the local dev backend) and may legitimately be
// written with a trailing slash, which would otherwise produce `http://host//chat`.
const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

// The market-data pathway (yfinance + matplotlib) has a measured P95 of ~59s, so /chat needs
// a far more generous ceiling than the rest of the API. Every request still gets *some*
// deadline so a dead backend can never leave the UI spinning forever.
const CHAT_TIMEOUT_MS = 180_000;
const INGEST_TIMEOUT_MS = 120_000;
const CLEAR_TIMEOUT_MS = 60_000;
const DEFAULT_TIMEOUT_MS = 30_000;

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
const ALLOWED_EXTENSIONS = ['.md', '.pdf'];

// Stable identity so `chats[key] || EMPTY_MESSAGES` doesn't hand every render a brand new
// array (which used to re-fire the auto-scroll effect on every single render).
const EMPTY_MESSAGES = [];

const GLOBAL_KEY = 'global';

const contextLabel = (key) => (key === GLOBAL_KEY ? 'Global Agent' : key);

/** Pulls a human-readable message out of a failed response. FastAPI reports errors as
 *  `{"detail": ...}`, where detail is a string for HTTPException and an array of objects for
 *  422 validation errors. Falls back to the status line so the user never sees a blank bubble. */
async function readErrorDetail(response) {
  let detail = '';
  try {
    const body = await response.json();
    if (typeof body?.detail === 'string') {
      detail = body.detail;
    } else if (Array.isArray(body?.detail)) {
      detail = body.detail.map((d) => d?.msg || JSON.stringify(d)).join('; ');
    } else if (typeof body?.message === 'string') {
      detail = body.message;
    } else if (body?.detail) {
      detail = JSON.stringify(body.detail);
    }
  } catch {
    // Body wasn't JSON (nginx HTML error page, empty 502, ...) — fall through to the status.
  }
  return detail || `The server responded with ${response.status} ${response.statusText || 'Error'}.`;
}

/** Single entry point for every backend call: absolute URL, hard timeout, caller-supplied
 *  cancellation, `response.ok` checking and JSON-shape validation. Throws an Error whose
 *  message is safe to show directly to the user. */
function createApiFetch(tokenRef) {
  return async function apiFetch(path, { timeoutMs = DEFAULT_TIMEOUT_MS, signal, ...init } = {}) {
    const controller = new AbortController();
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);

    const forwardAbort = () => controller.abort();
    if (signal) {
      if (signal.aborted) controller.abort();
      else signal.addEventListener('abort', forwardAbort);
    }

    // Inject auth header if a token is available
    const headers = { ...(init.headers || {}) };
    const currentToken = tokenRef.current;
    if (currentToken && !headers.Authorization) {
      headers.Authorization = `Bearer ${currentToken}`;
    }

    try {
      const response = await fetch(`${API_URL}${path}`, { ...init, headers, signal: controller.signal });

      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }

      try {
        return await response.json();
      } catch {
        throw new Error('The server returned a response that could not be read as JSON.');
      }
    } catch (error) {
      if (error?.name === 'AbortError') {
        if (timedOut) {
          const timeoutError = new Error(
            `The server did not respond within ${Math.round(timeoutMs / 1000)}s. It may still be working — try again in a moment.`
          );
          timeoutError.name = 'TimeoutError';
          throw timeoutError;
        }
        throw error; // Deliberate cancellation by the caller — handled upstream.
      }
      if (error instanceof TypeError) {
        // fetch() only rejects with TypeError for network/CORS level failures.
        throw new Error(`Could not reach the Finvisor backend at ${API_URL}. Is the API running?`);
      }
      throw error;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', forwardAbort);
    }
  };
}

const makeMessage = (role, content, extra = {}) => ({
  id: uuidv4(),
  role,
  content: typeof content === 'string' ? content : String(content ?? ''),
  ...extra,
});

/** react-markdown passes the raw mdast `node` to every override; forwarding it to a DOM
 *  element makes React log an unknown-prop warning. */
function stripNode(props) {
  const rest = { ...props };
  delete rest.node;
  return rest;
}

const markdownComponents = {
  // The backend is explicitly prompted to emit markdown tables. A wide table must scroll
  // inside its own bubble rather than stretching the flex layout past the viewport.
  table: (props) => (
    <div className="markdown-table-wrapper">
      <table {...stripNode(props)} />
    </div>
  ),
  a: (props) => <a {...stripNode(props)} target="_blank" rel="noopener noreferrer" />,
  pre: (props) => <pre className="markdown-pre" {...stripNode(props)} />,
};

/** Charts arrive as a base64 PNG that may be null (no chart) or corrupt (partial GridFS
 *  read). Both cases must degrade to a note instead of a broken-image icon. */
function ChartImage({ data }) {
  const [failed, setFailed] = useState(false);

  if (typeof data !== 'string' || data.length === 0) return null;

  if (failed) {
    return (
      <div className="chart-fallback">
        <ImageOff size={16} />
        <span>The generated chart could not be displayed.</span>
      </div>
    );
  }

  return (
    <img
      src={`data:image/png;base64,${data}`}
      alt="Chart generated by the agent"
      className="chart-image"
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}

function App() {
  const { user, token, isAuthenticated, loading: authLoading, logout } = useAuth();

  // Keep a ref to the token so apiFetch always reads the latest value without re-creating
  const tokenRef = useRef(token);
  useEffect(() => { tokenRef.current = token; }, [token]);

  const apiFetch = useMemo(() => createApiFetch(tokenRef), []);

  const [documents, setDocuments] = useState([]);
  const [documentsError, setDocumentsError] = useState(null);
  const [activeDocument, setActiveDocument] = useState(null); // null = Global Agent
  const [threads, setThreads] = useState(() => ({ [GLOBAL_KEY]: uuidv4() }));
  const [chats, setChats] = useState(() => ({ [GLOBAL_KEY]: [] }));

  const [inputValue, setInputValue] = useState('');
  const [pendingKey, setPendingKey] = useState(null); // chat key with an in-flight /chat call
  const [uploadStatus, setUploadStatus] = useState(null); // { text, tone }
  const [isUploading, setIsUploading] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [feedback, setFeedback] = useState({}); // messageId -> { rating, status, error }
  const [portfolio, setPortfolio] = useState([]);

  const chatEndRef = useRef(null);
  const inputRef = useRef(null);
  const chatAbortRef = useRef(null);
  // Requests whose result must be thrown away rather than appended (the conversation they
  // belonged to was reset while they were still in flight).
  const discardedRef = useRef(new WeakSet());
  const statusTimerRef = useRef(null);
  const isMountedRef = useRef(true);
  // Mirrors `threads` so a send can read/create a thread id synchronously instead of racing
  // the state update that `selectDocument` just queued.
  const threadsRef = useRef(threads);

  const docKey = activeDocument || GLOBAL_KEY;
  const currentMessages = chats[docKey] || EMPTY_MESSAGES;
  const isPending = pendingKey !== null;
  const messageCount = currentMessages.length;

  // Declared first so it is set back to true before any other effect runs on StrictMode's
  // deliberate double-mount.
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(
    () => () => {
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      chatAbortRef.current?.abort();
    },
    []
  );

  const globalPrompts = useMemo(
    () => [
      { label: '📊 Stock Chart', text: '📊 Plot a 6-month historical performance chart for ', requiresInput: true },
      { label: '🌐 Market News', text: '🌐 Fetch the latest macroeconomic market news and analyze current sentiment' },
      { label: '🧮 Stock Comparison', text: '🧮 Compare the fundamental metrics and valuations of ', requiresInput: true },
      { label: '📈 Intrinsic Value', text: '📈 Calculate the DCF intrinsic value for ', requiresInput: true },
      { label: '🎯 Sector Analysis', text: '🎯 Analyze the current market conditions and recent news for the following sector: ', requiresInput: true },
      { label: '💡 Market Trends', text: '💡 What are the top trending investment themes in the market right now?' },
    ],
    []
  );

  const docPrompts = useMemo(
    () => [
      { label: '📝 Executive Summary', text: '📝 Provide a comprehensive executive summary of this report, highlighting the most critical strategic insights.' },
      { label: '⚠️ Risk Analysis', text: '⚠️ Identify and analyze the key operational and financial risks mentioned in this document.' },
      { label: '📊 Stock Chart', text: "📊 Extract the company's ticker and plot a 6-month historical performance chart." },
      { label: '🧮 Financial Calculators', text: '🧮 Extract the key financial metrics from this document and calculate the profit margin and year-over-year growth.' },
      { label: '🌐 Market News', text: '🌐 Fetch the latest macroeconomic news relevant to the company or sector discussed in this report.' },
      { label: '🎯 Sentiment Analysis', text: '🎯 Analyze the overall tone and sentiment of this report. Is management optimistic or cautious?' },
    ],
    []
  );

  const currentAvailablePrompts = useMemo(
    () =>
      (activeDocument ? docPrompts : globalPrompts).filter(
        (p) => !currentMessages.some((m) => m.role === 'user' && m.content === p.text)
      ),
    [activeDocument, docPrompts, globalPrompts, currentMessages]
  );

  // Depends on the message *count*, not the array identity, so it fires once per new message
  // instead of on every render.
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messageCount, isPending, activeDocument]);

  const writeThreads = useCallback((updater) => {
    const next = updater(threadsRef.current);
    threadsRef.current = next;
    setThreads(next);
  }, []);

  /** Guarantees a stable thread id per chat context. The backend's LangGraph checkpointer
   *  keys conversation memory on this value, so it must never be undefined (FastAPI would
   *  422) and must not change mid-conversation. */
  const getThreadId = useCallback(
    (key) => {
      const existing = threadsRef.current[key];
      if (existing) return existing;
      const created = uuidv4();
      writeThreads((prev) => ({ ...prev, [key]: created }));
      return created;
    },
    [writeThreads]
  );

  const showStatus = useCallback((text, tone = 'info', autoClearMs = 6000) => {
    if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
    statusTimerRef.current = null;
    setUploadStatus({ text, tone });
    if (autoClearMs > 0) {
      statusTimerRef.current = setTimeout(() => {
        statusTimerRef.current = null;
        if (isMountedRef.current) setUploadStatus(null);
      }, autoClearMs);
    }
  }, []);

  const appendMessage = useCallback((key, message) => {
    setChats((prev) => ({ ...prev, [key]: [...(prev[key] || []), message] }));
  }, []);

  const fetchDocuments = useCallback(async (signal) => {
    try {
      const data = await apiFetch('/documents', { signal });
      if (!isMountedRef.current) return;
      const list = Array.isArray(data?.documents) ? data.documents : [];
      // The backend derives this from Chroma metadata, which can contain duplicates and
      // null sources — both would produce blank rows / duplicate React keys.
      const clean = [...new Set(list.filter((d) => typeof d === 'string' && d.trim()))].sort((a, b) =>
        a.localeCompare(b)
      );
      setDocuments(clean);
      setDocumentsError(null);
    } catch (error) {
      if (error?.name === 'AbortError' || !isMountedRef.current) return;
      setDocumentsError(error?.message || 'Could not load the document list.');
    }
  }, [apiFetch]);

  // Fetch portfolio on login
  const fetchPortfolio = useCallback(async () => {
    if (!isAuthenticated) return;
    try {
      const data = await apiFetch('/portfolio');
      if (isMountedRef.current) {
        setPortfolio(data?.tickers || []);
      }
    } catch {
      // Portfolio fetch failure is non-critical
    }
  }, [isAuthenticated, apiFetch]);

  // Load chat history for a context from the backend
  const loadChatHistory = useCallback(async (contextKey) => {
    if (!isAuthenticated) return;
    try {
      const data = await apiFetch(`/chat-history/${encodeURIComponent(contextKey)}`);
      if (!isMountedRef.current) return;
      const messages = data?.messages || [];
      if (messages.length > 0) {
        const restored = messages.map((m) =>
          makeMessage(m.role, m.content, {
            ...(m.extra || {}),
            // Mark as restored so we don't re-persist
            restored: true,
          })
        );
        setChats((prev) => {
          // Only restore if the current chat is empty (don't overwrite live conversation)
          if (!prev[contextKey] || prev[contextKey].length === 0) {
            return { ...prev, [contextKey]: restored };
          }
          return prev;
        });
      }
    } catch {
      // History fetch failure is non-critical
    }
  }, [isAuthenticated, apiFetch]);

  useEffect(() => {
    if (!isAuthenticated) return;
    const controller = new AbortController();
    fetchDocuments(controller.signal);
    fetchPortfolio();
    loadChatHistory(GLOBAL_KEY);
    return () => controller.abort();
  }, [isAuthenticated, fetchDocuments, fetchPortfolio, loadChatHistory]);

  const handleCancelRequest = useCallback(() => {
    chatAbortRef.current?.abort();
  }, []);

  const handleSend = useCallback(
    async (textOverride) => {
      const raw = typeof textOverride === 'string' ? textOverride : inputValue;
      const userPrompt = raw.trim();
      if (!userPrompt || chatAbortRef.current) return;

      const key = activeDocument || GLOBAL_KEY;
      const sourceFilename = activeDocument;
      const threadId = getThreadId(key);

      if (typeof textOverride !== 'string') setInputValue('');

      appendMessage(key, makeMessage('user', userPrompt));
      setPendingKey(key);

      const controller = new AbortController();
      chatAbortRef.current = controller;

      try {
        const payload = { prompt: userPrompt, thread_id: threadId, context_key: key };
        if (sourceFilename) payload.source_filename = sourceFilename;

        const data = await apiFetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          timeoutMs: CHAT_TIMEOUT_MS,
          signal: controller.signal,
        });

        if (!isMountedRef.current || discardedRef.current.has(controller)) return;

        const answer =
          typeof data?.response === 'string' && data.response.trim()
            ? data.response
            : '_The agent finished but returned an empty response._';

        appendMessage(
          key,
          makeMessage('agent', answer, {
            chart_base64: typeof data?.chart_base64 === 'string' ? data.chart_base64 : null,
            question: userPrompt,
            threadId,
            sourceFilename,
          })
        );
      } catch (error) {
        if (!isMountedRef.current || discardedRef.current.has(controller)) return;
        const isCancel = error?.name === 'AbortError';
        appendMessage(
          key,
          makeMessage(
            'agent',
            isCancel ? 'Request cancelled.' : error?.message || 'Something went wrong talking to the backend.',
            { isError: true }
          )
        );
      } finally {
        if (chatAbortRef.current === controller) chatAbortRef.current = null;
        if (isMountedRef.current) setPendingKey(null);
      }
    },
    [activeDocument, appendMessage, getThreadId, inputValue, apiFetch]
  );

  const handleFeedback = useCallback(
    async (message, rating) => {
      const current = feedback[message.id];
      if (current?.status === 'sending' || (current?.status === 'sent' && current.rating === rating)) return;

      setFeedback((prev) => ({ ...prev, [message.id]: { rating, status: 'sending' } }));

      try {
        await apiFetch('/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            thread_id: message.threadId,
            source_filename: message.sourceFilename || null,
            question: message.question || '',
            answer: message.content,
            rating,
          }),
        });
        if (!isMountedRef.current) return;
        setFeedback((prev) => ({ ...prev, [message.id]: { rating, status: 'sent' } }));
      } catch (error) {
        if (!isMountedRef.current) return;
        setFeedback((prev) => ({
          ...prev,
          [message.id]: { rating, status: 'error', error: error?.message || 'Could not send feedback.' },
        }));
      }
    },
    [feedback, apiFetch]
  );

  const handleFileUpload = useCallback(
    async (event) => {
      const input = event.target;
      const file = input.files?.[0];
      // Clear immediately so re-selecting the *same* file still fires onChange, and so a
      // failed upload can be retried without picking a different file first.
      input.value = '';
      if (!file) return;

      const dot = file.name.lastIndexOf('.');
      const extension = dot === -1 ? '' : file.name.slice(dot).toLowerCase();

      if (!ALLOWED_EXTENSIONS.includes(extension)) {
        showStatus(`Unsupported file type "${extension || file.name}". Upload a .md or .pdf report.`, 'error');
        return;
      }
      if (file.size === 0) {
        showStatus(`"${file.name}" is empty — nothing to ingest.`, 'error');
        return;
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        showStatus(
          `"${file.name}" is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is ${MAX_UPLOAD_BYTES / 1024 / 1024} MB.`,
          'error'
        );
        return;
      }

      setIsUploading(true);
      showStatus(`Chunking and embedding ${file.name}…`, 'info', 0);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const data = await apiFetch('/ingest', {
          method: 'POST',
          body: formData,
          timeoutMs: INGEST_TIMEOUT_MS,
        });
        if (!isMountedRef.current) return;
        const chunks = Number.isFinite(data?.chunk_count) ? ` (${data.chunk_count} chunks)` : '';
        showStatus(`${data?.message || `Successfully ingested ${file.name}`}${chunks}`, 'success');
        fetchDocuments();
      } catch (error) {
        if (!isMountedRef.current) return;
        showStatus(error?.message || `Could not ingest ${file.name}.`, 'error');
      } finally {
        if (isMountedRef.current) setIsUploading(false);
      }
    },
    [fetchDocuments, showStatus, apiFetch]
  );

  const handleClearDatabase = useCallback(async () => {
    if (isClearing) return;
    if (!window.confirm('Are you sure you want to clear all uploaded documents? This cannot be undone.')) return;

    setIsClearing(true);
    showStatus('Clearing database…', 'info', 0);

    try {
      const data = await apiFetch('/documents', { method: 'DELETE', timeoutMs: CLEAR_TIMEOUT_MS });
      if (!isMountedRef.current) return;

      setDocuments([]);
      setActiveDocument(null);
      setDocumentsError(null);
      // Drop the per-document threads and transcripts: those documents no longer exist, so
      // the checkpointer memory behind those thread ids now refers to deleted context.
      writeThreads((prev) => ({ [GLOBAL_KEY]: prev[GLOBAL_KEY] || uuidv4() }));
      setChats((prev) => ({ [GLOBAL_KEY]: prev[GLOBAL_KEY] || [] }));
      showStatus(data?.message || 'Database cleared successfully!', 'success');
    } catch (error) {
      if (!isMountedRef.current) return;
      showStatus(error?.message || 'Could not clear the database.', 'error');
    } finally {
      if (isMountedRef.current) setIsClearing(false);
    }
  }, [isClearing, showStatus, writeThreads, apiFetch]);



  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const selectDocument = useCallback(
    (doc) => {
      setActiveDocument(doc);
      const key = doc || GLOBAL_KEY;
      getThreadId(key);
      setChats((prev) => (prev[key] ? prev : { ...prev, [key]: [] }));
      // Load persisted history for this context
      loadChatHistory(key);
    },
    [getThreadId, loadChatHistory]
  );

  const handleDocKeyDown = useCallback(
    (e, doc) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        selectDocument(doc);
      }
    },
    [selectDocument]
  );

  const handleDeleteDocument = useCallback(
    async (e, doc) => {
      e.stopPropagation(); // Don't trigger selectDocument when clicking the trash can
      if (!window.confirm(`Are you sure you want to delete "${doc}"?`)) return;

      try {
        const data = await apiFetch(`/documents/${encodeURIComponent(doc)}`, { method: 'DELETE', timeoutMs: CLEAR_TIMEOUT_MS });
        if (!isMountedRef.current) return;
        
        showStatus(data?.message || `Deleted ${doc}`, 'success');
        
        // If we just deleted the active document, fall back to global agent
        if (activeDocument === doc) {
          selectDocument(null);
        }
        
        // Remove from local state and trigger a refresh from the server just to be perfectly synced
        setDocuments((prev) => prev.filter((d) => d !== doc));
      } catch (error) {
        if (!isMountedRef.current) return;
        showStatus(error?.message || `Could not delete ${doc}.`, 'error');
      }
    },
    [activeDocument, selectDocument, showStatus, apiFetch]
  );

  /** Starts a genuinely fresh conversation: new thread id (so the backend checkpointer starts
   *  from an empty state) and an empty transcript for the active context only. */
  const handleNewChat = useCallback(async () => {
    const inFlight = chatAbortRef.current;
    if (inFlight) {
      // Suppress the reply/cancellation notice so it can't land in the freshly emptied chat.
      discardedRef.current.add(inFlight);
      inFlight.abort();
    }
    const key = activeDocument || GLOBAL_KEY;
    writeThreads((prev) => ({ ...prev, [key]: uuidv4() }));
    setChats((prev) => ({ ...prev, [key]: [] }));

    // Also clear persisted history for this context
    if (isAuthenticated) {
      try {
        await apiFetch(`/chat-history/${encodeURIComponent(key)}`, { method: 'DELETE' });
      } catch {
        // Non-critical
      }
    }

    inputRef.current?.focus();
  }, [activeDocument, writeThreads, isAuthenticated, apiFetch]);

  const handleTickerClick = useCallback(
    (ticker) => {
      const prompt = `📊 Plot a 6-month historical performance chart for ${ticker}`;
      const userMessageIndex = currentMessages.findIndex(
        (m) => m.role === 'user' && m.content === prompt
      );

      if (userMessageIndex !== -1) {
        showStatus(`A chart for ${ticker} has already been plotted in this conversation.`, 'info');
        // The actual chart is the agent's response, which is usually the very next message
        const targetMessage = currentMessages[userMessageIndex + 1] || currentMessages[userMessageIndex];
        const element = document.getElementById(`message-${targetMessage.id}`);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'center' });
          element.animate(
            [
              { backgroundColor: 'rgba(59, 130, 246, 0.4)' },
              { backgroundColor: 'transparent' }
            ],
            { duration: 1500, easing: 'ease-out' }
          );
        }
        return;
      }

      handleSend(prompt);
    },
    [handleSend, currentMessages, showStatus]
  );

  // Show loading spinner while checking stored token
  if (authLoading) {
    return (
      <div className="auth-loading">
        <span className="loader" style={{ width: '40px', height: '40px', borderWidth: '3px' }} />
        <p>Loading Finvisor AI…</p>
      </div>
    );
  }

  // Show auth page if not logged in
  if (!isAuthenticated) {
    return <AuthPage />;
  }

  const inputPlaceholder = !isPending
    ? 'Type your message here...'
    : pendingKey === docKey
      ? 'Waiting for the agent…'
      : `Waiting for a response in ${contextLabel(pendingKey)}…`;

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="glass-panel sidebar">
        <h2>📈 Finvisor AI</h2>

        {/* User profile */}
        <div className="user-profile">
          <div className="user-avatar">
            <User size={18} />
          </div>
          <div className="user-info">
            <span className="user-name">{user?.username || 'User'}</span>
            <span className="user-email">{user?.email || ''}</span>
          </div>
          <button
            type="button"
            className="logout-btn"
            onClick={logout}
            title="Sign out"
            aria-label="Sign out"
          >
            <LogOut size={16} />
          </button>
        </div>

        <label className={`file-upload-box${isUploading ? ' is-disabled' : ''}`}>
          <input
            type="file"
            className="file-input"
            accept=".md,.pdf"
            onChange={handleFileUpload}
            disabled={isUploading}
          />
          <UploadCloud size={24} style={{ color: 'var(--accent-color)', flexShrink: 0 }} />
          <div style={{ textAlign: 'left', minWidth: 0 }}>
            <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.95rem' }}>
              {isUploading ? 'Uploading…' : 'Upload Document'}
            </div>
            <div style={{ fontSize: '0.75rem', marginTop: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              Max 25 MB (.pdf, .md)
            </div>
          </div>
        </label>

        {isUploading && (
          <div style={{ textAlign: 'center', marginBottom: '16px' }}>
            <span className="loader"></span>
          </div>
        )}
        {uploadStatus && (
          <div className={`upload-status upload-status-${uploadStatus.tone}`} role="status">
            {uploadStatus.text}
          </div>
        )}

        {/* Portfolio */}
        <PortfolioPanel
          token={token}
          tickers={portfolio}
          onTickersChange={setPortfolio}
          onTickerClick={handleTickerClick}
        />

        <div className="document-list">
          <h3
            style={{
              fontSize: '0.9rem',
              color: 'var(--text-secondary)',
              marginBottom: '8px',
              textTransform: 'uppercase',
            }}
          >
            Chat Contexts
          </h3>
          <div
            className={`doc-item ${activeDocument === null ? 'active' : ''}`}
            onClick={() => selectDocument(null)}
            onKeyDown={(e) => handleDocKeyDown(e, null)}
            role="button"
            tabIndex={0}
          >
            <Globe size={16} style={{ marginRight: '8px' }} />
            <span>Global Agent</span>
          </div>
          {documents.map((doc) => (
            <div
              key={doc}
              className={`doc-item ${activeDocument === doc ? 'active' : ''}`}
              onClick={() => selectDocument(doc)}
              onKeyDown={(e) => handleDocKeyDown(e, doc)}
              role="button"
              tabIndex={0}
              title={doc}
            >
              <Database size={16} style={{ marginRight: '8px', flexShrink: 0 }} />
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>{doc}</span>
              <button
                type="button"
                className="doc-delete-btn"
                onClick={(e) => handleDeleteDocument(e, doc)}
                title="Delete this document"
                aria-label="Delete document"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}

          {documentsError && (
            <div className="sidebar-error">
              <span>{documentsError}</span>
              <button type="button" className="link-btn" onClick={() => fetchDocuments()}>
                <RefreshCw size={13} />
                Retry
              </button>
            </div>
          )}
        </div>

        <div style={{ marginTop: 'auto', paddingTop: '24px' }}>
          <button type="button" className="danger-btn" onClick={handleClearDatabase} disabled={isClearing}>
            <Trash2 size={16} />
            {isClearing ? 'Clearing…' : 'Clear Database'}
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="glass-panel chat-container">
        <div
          className="chat-header"
          style={{
            padding: '16px 24px',
            borderBottom: '1px solid rgba(255,255,255,0.1)',
            display: 'flex',
            alignItems: 'center',
            gap: '16px',
          }}
        >
          <h2 style={{ fontSize: '1.2rem', margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {activeDocument ? `Chatting with: ${activeDocument}` : 'Global Agent'}
          </h2>
          <button
            type="button"
            className="new-chat-btn"
            onClick={handleNewChat}
            disabled={messageCount === 0 && pendingKey !== docKey}
            title="Start a fresh conversation in this context"
          >
            <SquarePen size={15} />
            New Chat
          </button>
        </div>
        <div className="chat-history" style={{ padding: '24px' }}>
          {messageCount === 0 && (
            <div style={{ margin: 'auto', textAlign: 'center', color: 'var(--text-secondary)' }}>
              <h3 style={{ fontSize: '1.8rem', marginBottom: '12px', color: 'var(--text-primary)' }}>
                {activeDocument ? `Analyze ${activeDocument}` : `Welcome back, ${user?.username || 'User'}!`}
              </h3>
              <p style={{ marginBottom: '32px', fontSize: '1.2rem' }}>
                {activeDocument
                  ? `Ask questions to extract insights exclusively from ${activeDocument}.`
                  : 'Analyze live market trends, compare stock fundamentals, or calculate intrinsic valuations across any equity.'}
              </p>
            </div>
          )}

          {currentMessages.map((msg) => {
            const state = feedback[msg.id];
            return (
              <div
                key={msg.id}
                id={`message-${msg.id}`}
                className={`message-bubble message-${msg.role}${msg.isError ? ' message-error' : ''}`}
              >
                {msg.role === 'user' ? (
                  msg.content
                ) : (
                  <>
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                      {msg.content}
                    </ReactMarkdown>
                    <ChartImage data={msg.chart_base64} />
                    {!msg.isError && msg.threadId && (
                      <div className="feedback-row">
                        <button
                          type="button"
                          className={`feedback-btn${state?.rating === 'up' && state?.status !== 'error' ? ' active' : ''}`}
                          onClick={() => handleFeedback(msg, 'up')}
                          disabled={state?.status === 'sending'}
                          title="This answer was helpful"
                          aria-label="This answer was helpful"
                        >
                          <ThumbsUp size={14} />
                        </button>
                        <button
                          type="button"
                          className={`feedback-btn${state?.rating === 'down' && state?.status !== 'error' ? ' active down' : ''}`}
                          onClick={() => handleFeedback(msg, 'down')}
                          disabled={state?.status === 'sending'}
                          title="This answer was not helpful"
                          aria-label="This answer was not helpful"
                        >
                          <ThumbsDown size={14} />
                        </button>
                        {state?.status === 'sent' && <span className="feedback-note">Thanks for the feedback</span>}
                        {state?.status === 'error' && (
                          <span className="feedback-note feedback-note-error">{state.error}</span>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            );
          })}

          {isPending && pendingKey === docKey && (
            <div className="thinking-indicator">
              <span className="loader" style={{ width: '16px', height: '16px', borderWidth: '2px' }}></span>
              Agent is thinking...
              <button type="button" className="link-btn" onClick={handleCancelRequest}>
                Cancel
              </button>
            </div>
          )}

          {!isPending && currentAvailablePrompts.length > 0 && (
            <div style={{ marginTop: messageCount === 0 ? '0' : '24px' }}>
              <p
                style={{
                  color: 'var(--text-secondary)',
                  fontSize: messageCount === 0 ? '1.2rem' : '1rem',
                  marginBottom: '16px',
                  textAlign: messageCount === 0 ? 'center' : 'left',
                }}
              >
                {messageCount === 0
                  ? 'OR Select from the given buttons below!'
                  : 'Click on the following to explore more:'}
              </p>
              <div className="quick-prompts" style={{ justifyContent: messageCount === 0 ? 'center' : 'flex-start' }}>
                {currentAvailablePrompts.map((p) => (
                  <button
                    type="button"
                    key={p.label}
                    className="prompt-chip"
                    onClick={() => {
                      if (p.requiresInput) {
                        setInputValue(p.text);
                        inputRef.current?.focus();
                      } else {
                        handleSend(p.text);
                      }
                    }}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <div className="input-container">
          <input
            ref={inputRef}
            type="text"
            className="chat-input"
            placeholder={inputPlaceholder}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isPending}
          />
          <button
            type="button"
            className={`send-btn${isPending ? ' stop-btn' : ''}`}
            onClick={isPending ? handleCancelRequest : () => handleSend()}
            disabled={!isPending && !inputValue.trim()}
            title={isPending ? 'Cancel the current request' : 'Send message'}
            aria-label={isPending ? 'Cancel the current request' : 'Send message'}
          >
            {isPending ? <CircleStop size={20} /> : <Send size={20} />}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
