import { flagBubble, isDismissed } from './banner.js';

const MSG_CONTAINER_SEL = '[data-testid="msg-container"]';
const TEXT_SEL = 'span[data-testid="selectable-text"]';
const MESSAGE_TEXT_SEL = `${MSG_CONTAINER_SEL} ${TEXT_SEL}`;
const CONTACT_SEL = '[data-testid="conversation-info-header-chat-title"]';
const PANE_SELECTORS = ['#main', '[data-testid="conversation-panel-messages"]'];


// whatsapp rerenders a message several times (sent->delivered->read), so 
// wait a moment and only handle each element once 
const DEBOUNCE_MS = 75;
const MIN_TEXT_LENGTH = 5;
const MAX_RETRIES = 5;

// per session memory : with key as "contact + text"
const results = new Map(); // key -> result 
const inFlight = new Set(); // waiting for service worker !
const scheduled = new WeakSet(); // elements waiting for the debounce timer


const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function getContactName() {
  return document.querySelector(CONTACT_SEL)?.textContent?.trim() || 'Unknown chat';
}

// outgoing only ( our own mesages are not scanned )
function isOutgoing(el) {
  if (el.closest('.message-out')) return true;
  if (el.closest('.message-in')) return false;

  // Fallback: only our own messages show send-status icons (ticks, clock, ...).
  const container = el.closest(MSG_CONTAINER_SEL);
  if (!container) return false;
  const outgoingIcons = ['tail-out', 'msg-check', 'msg-dblcheck', 'msg-time', 'msg-error'];
  return [...container.querySelectorAll('[data-icon]')].some(
    (icon) =>
      !icon.closest('[data-testid="quoted-message"]') && outgoingIcons.includes(icon.dataset.icon),
  );
}

// --- Talking to the service worker ------------------------------------------

function sendToWorker(text, contact) {
  return new Promise((resolve) => { // promise hai, when returned we would call back this resolve function with teh correct parameters 
    try {
      chrome.runtime.sendMessage({ type: 'ANALYZE', text, contact }, (response) => {
        const error = chrome.runtime.lastError?.message;
        resolve(error ? { error } : response || { error: 'Empty worker response.' });
      });
    } catch (error) {
      resolve({ error: String(error?.message || error) }); // e.g. extension was reloaded
    }
  });
}


async function classifyWithRetry(text, contact) {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const response = await sendToWorker(text, contact);
    if (!response.error) return response;
    await sleep(Math.min(8000, 500 * 2 ** attempt));
  }
  return null; 
}

// --- Message handling -------------------------------------------------------

/** Schedule an element for scanning (debounced). */
function scan(el) {
  if (scheduled.has(el)) return;
  scheduled.add(el);
  setTimeout(() => {
    scheduled.delete(el);
    handleMessage(el);
  }, DEBOUNCE_MS);
}

async function handleMessage(el) {
  if (!el.isConnected || isOutgoing(el)) return; // is it properly attached to the DOM tree 

  const text = el.innerText?.trim();
  if (!text || text.length < MIN_TEXT_LENGTH) return;

  const contact = getContactName();
  const key = `${contact}\n${text}`;
  if (isDismissed(key)) return;

  // Already classified: just re-show the banner if WhatsApp re-rendered the bubble.
  if (results.has(key)) {
    const flagged = results.get(key);
    if (flagged) flagBubble(el, { ...flagged, id: key });
    return;
  }

  if (inFlight.has(key)) return;
  inFlight.add(key);

  const response = await classifyWithRetry(text, contact);
  inFlight.delete(key);
  if (!response) return;

  const flagged = response.flagged ? { score: response.score, severity: response.severity} : null;
  results.set(key, flagged);
  if (flagged && el.isConnected) flagBubble(el, { ...flagged, id: key });
}

// --- Watching the chat pane -------------------------------------------------

/** Scan every message text near a changed DOM node. */
function scanAround(node) {
  const el = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
  if (!el) return;
  // the node is inside a message (e.g. ticks changed): rescan that message
  el.closest(MSG_CONTAINER_SEL)?.querySelectorAll(TEXT_SEL).forEach(scan);
  // queryselectorALl returns all relevant similar messages !
  // the node contains messages (e.g. new messages were added)
  el.querySelectorAll(MESSAGE_TEXT_SEL).forEach(scan);
}

let currentPane = null;
let paneObserver = null;
let attachTimer = null;

function attachSoon(delay) {
  if (attachTimer) return;
  attachTimer = setTimeout(() => {
    attachTimer = null; // running of this fn means the code has executed hence the global variable attachTimer is set to null so that new timer could be attached if needed 
    attach();
  }, delay);
}

function attach() {
  const pane = PANE_SELECTORS.map((sel) => document.querySelector(sel)).find(Boolean); // first true value 
  if (!pane) return attachSoon(500); // WhatsApp hasn't rendered the chat yet
  if (pane === currentPane) return;

  paneObserver?.disconnect();
  currentPane = pane;

  pane.querySelectorAll(MESSAGE_TEXT_SEL).forEach(scan); // messages already on screen

  paneObserver = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      mutation.addedNodes.forEach(scanAround); // all the addedNodes (if multiple nodes are added )
      scanAround(mutation.target); // exact location of mutation 
    }
  });
  paneObserver.observe(pane, {
    childList: true,
    subtree: true,
    characterData: true,
    attributes: true,
    attributeFilter: ['class', 'data-icon'],
  });
}

// If the pane we were watching disappears, find the new one.
new MutationObserver(() => {
  if (!currentPane?.isConnected) attachSoon(250);
}).observe(document.documentElement, { childList: true, subtree: true });

attach();
