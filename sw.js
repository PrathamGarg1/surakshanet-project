import { classify, loadModel } from './classifier.js';
import { saveIncident } from './store.js';

const jobs = [];
let isBusy = false;

async function workThroughJobs() {
  if (isBusy) return;
  isBusy = true;

  try {
    while (jobs.length > 0) {
      const job = jobs.shift();
      try {
        const result = await classify(job.text);
        if (result.flagged) {
          await saveIncident({ text: job.text, contact: job.contact, ...result });
        }
        job.reply(result); 
        // reply is the same callback called from the 
        // content script 
        // ie "sendResponse" 
      } catch (error) {
        // A failed message must not stop the ones waiting behind it.
        job.reply({ flagged: false, error: String(error.message || error) });
      }
    }
  } finally {
    isBusy = false;
    if (jobs.length > 0) workThroughJobs();
  }
}

async function preloadModel() {
  try {
    await loadModel();
  } catch {
    // The next real message will try to load the model again.
  }
}

chrome.runtime.onInstalled.addListener(() => {
  preloadModel();
});
chrome.runtime.onStartup.addListener(() => {
  preloadModel();
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  // _sender represnets the TAB which sent it 
  // sendResponse is the callback sent be chrome to be called when listened!!

  if (msg?.type !== 'ANALYZE') return false;

  if (typeof msg.text !== 'string' || !msg.text.trim()) {
    sendResponse({ flagged: false, error: 'Message text is required.' });
    return false;
  }

  jobs.push({ text: msg.text, contact: msg.contact, reply: sendResponse });
  workThroughJobs();

  // Chrome closes the channel unless we return true before the awaits finish.
  return true;
});
