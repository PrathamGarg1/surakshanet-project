/**
 * Warning UI: highlights a flagged message bubble and adds a banner below it.
 */

const STYLE_ID = 'suraksha-banner-styles';

const SEVERITY = {
  high: { label: 'Critical threat', border: '#d32f2f', background: '#ffebee', text: '#8f1d1d' },
  medium: { label: 'Harmful content', border: '#f57c00', background: '#fff3e0', text: '#9a4d00' },
};

// Messages the user dismissed (kept in memory only, cleared on page reload).
const dismissed = new Set();

export function isDismissed(id) {
  return Boolean(id && dismissed.has(id));
}

// Colors come from CSS variables (--sn-bg, --sn-border, --sn-text) set per banner.
const BANNER_CSS = `
  .suraksha-banner {
    align-items: center !important;
    box-sizing: border-box !important;
    display: flex !important;
    gap: 9px !important;
    float: none !important;
    margin: 7px 9px 5px !important;
    min-height: 51px !important;
    padding: 8px 9px !important;
    border: 1px solid color-mix(in srgb, var(--sn-border) 30%, transparent) !important;
    border-left: 4px solid var(--sn-border) !important;
    border-radius: 10px !important;
    color: var(--sn-text) !important;
    background: var(--sn-bg) !important;
    box-shadow: 0 5px 14px rgba(0, 0, 0, .16) !important;
    font: 500 12px/1.2 "Avenir Next", "Segoe UI", sans-serif !important;
  }
  .suraksha-banner__icon {
    align-items: center !important;
    background: rgba(255, 255, 255, .7) !important;
    border-radius: 7px !important;
    color: var(--sn-border) !important;
    display: flex !important;
    flex: 0 0 27px !important;
    height: 27px !important;
    justify-content: center !important;
  }
  .suraksha-banner__copy {
    display: grid !important;
    flex: 1 1 auto !important;
    gap: 2px !important;
    min-width: 0 !important;
  }
  .suraksha-banner__title {
    font-size: 11px !important;
    font-weight: 700 !important;
  }
  .suraksha-banner__detail {
    font-size: 10px !important;
    opacity: .75 !important;
  }
  .suraksha-banner__dismiss {
    min-height: 30px !important;
    padding: 5px 8px !important;
    border: 1px solid color-mix(in srgb, var(--sn-border) 48%, transparent) !important;
    border-radius: 7px !important;
    color: var(--sn-text) !important;
    background: transparent !important;
    cursor: pointer !important;
    font: 600 10px/1 "Avenir Next", "Segoe UI", sans-serif !important;
  }
  .suraksha-banner__dismiss:hover {
    background: color-mix(in srgb, var(--sn-border) 10%, transparent) !important;
  }
`;

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = BANNER_CSS;
  (document.head || document.documentElement).appendChild(style);
}

function makeShieldIcon() {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 24 26');
  svg.setAttribute('width', '16');
  svg.setAttribute('height', '18');
  svg.setAttribute('fill', 'none');

  const addPath = (d, extra = {}) => {
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', d);
    path.setAttribute('stroke', 'currentColor');
    path.setAttribute('stroke-width', '1.7');
    Object.entries(extra).forEach(([name, value]) => path.setAttribute(name, value));
    svg.appendChild(path);
  };

  addPath('M12 2.5 21 6v6.4c0 5.4-3.5 9.3-9 11.1-5.5-1.8-9-5.7-9-11.1V6l9-3.5Z'); // shield
  addPath('m8.2 13 2.5 2.5 5.3-5.4', { 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }); // tick
  return svg;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text) node.textContent = text;
  return node;
}

/** Places the banner under the bubble, aligned to the bubble's side of the chat. */
function alignBanner(bar, bubble, bubbleParent) {
  if (!bubble || !bubbleParent) return;
  const bubbleRect = bubble.getBoundingClientRect();
  const parentRect = bubbleParent.getBoundingClientRect();
  const isRightAligned = bubbleRect.left + bubbleRect.width / 2 > window.innerWidth / 2;
  const width = Math.min(360, Math.max(220, bubbleRect.width));
  const left = isRightAligned
    ? bubbleRect.right - width - parentRect.left
    : bubbleRect.left - parentRect.left;

  bar.style.setProperty('margin-left', `${Math.max(9, left)}px`, 'important');
  bar.style.setProperty('margin-right', '0px', 'important');
  bar.style.setProperty('width', `${width}px`, 'important');
}

/**
 * Highlights the bubble that contains `textEl` and adds a warning banner.
 * `id` identifies the message so a dismissal is remembered.
 */
export function flagBubble(textEl, { score, severity, id }) {
  const config = SEVERITY[severity] || SEVERITY.medium;
  const bubble = textEl.closest('[data-testid="msg-container"]') || textEl.parentElement;
  const bubbleParent = bubble?.parentElement;
  const messageRow = textEl.closest('[data-testid^="conv-msg-"]') || bubbleParent || bubble;

  if (messageRow?.querySelector('.suraksha-banner')) return; // already shown
  ensureStyles();

  // Highlight the bubble.
  bubble?.style.setProperty('border', `2px solid ${config.border}`, 'important');
  bubble?.style.setProperty('border-radius', '10px', 'important');
  bubble?.style.setProperty('background-color', `${config.border}12`, 'important');

  // Build the banner.
  const bar = el('div', 'suraksha-banner');
  bar.setAttribute('role', 'status');
  bar.style.setProperty('--sn-bg', config.background);
  bar.style.setProperty('--sn-border', config.border);
  bar.style.setProperty('--sn-text', config.text);
  alignBanner(bar, bubble, bubbleParent);

  const icon = el('span', 'suraksha-banner__icon');
  icon.setAttribute('aria-hidden', 'true');
  icon.appendChild(makeShieldIcon());

  const copy = el('span', 'suraksha-banner__copy');
  copy.append(
    el('strong', 'suraksha-banner__title', config.label),
    el('span', 'suraksha-banner__detail', `${Math.round(score * 100)}% confidence · saved locally`),
  );

  const dismissButton = el('button', 'suraksha-banner__dismiss', 'Dismiss');
  dismissButton.type = 'button';
  dismissButton.setAttribute('aria-label', 'Dismiss this SurakshaNet warning');
  dismissButton.onclick = (event) => {
    event.stopPropagation();
    if (id) dismissed.add(id);
    ['border', 'border-radius', 'background-color'].forEach((p) => bubble?.style.removeProperty(p));
    bar.remove();
  };

  bar.append(icon, copy, dismissButton);
  bubbleParent?.appendChild(bar);
}
