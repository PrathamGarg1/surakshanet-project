/** Local incident log (chrome.storage.local). Nothing leaves the device. */
const KEY = 'surakshanet_incidents';
const MAX_INCIDENTS = 200; // oldest ones are dropped beyond this

async function sha256(text) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export async function listIncidents() {
  const { [KEY]: list = [] } = await chrome.storage.local.get(KEY);
  return list;
}

/** Saves a flagged message. Returns the new incident, or null if it was a duplicate. */
export async function saveIncident({ text, contact, score, severity }) {
  const chat = typeof contact === 'string' && contact.trim() ? contact.trim() : 'Unknown chat';
  const id = await sha256(`${chat}\n${text.trim().toLowerCase()}`);

  const list = await listIncidents();
  if (list.some((incident) => incident.id === id)) return null;

  const incident = { id, ts: new Date().toISOString(), contact: chat, score, severity, text };
  await chrome.storage.local.set({ [KEY]: [incident, ...list].slice(0, MAX_INCIDENTS) });
  return incident;
}

export async function clearIncidents() {
  await chrome.storage.local.set({ [KEY]: [] });
}

export async function exportJSON() {
  const incidents = await listIncidents();
  return JSON.stringify({ exportedAt: new Date().toISOString(), incidents }, null, 2);
}
