# SurakshaNet

SurakshaNet is a Chrome Manifest V3 extension that detects abusive incoming
WhatsApp Web messages locally in the browser. Message text is never sent to a
server.

## Demo

[Watch the video demo on Google Drive](https://drive.google.com/file/d/1ElkdF0XA2bNzUCL6VWI95M_fqqFI4mmR/view?usp=sharing)

## Features

- On-device Transformers.js classification for incoming messages
- Contact-aware incident log with local SHA-256 deduplication
- Resilient WhatsApp DOM scanning and banner reattachment
- JSON and PDF export, including Hindi/Devanagari text
- No cloud sync, reporting, or automatic blocking

## Requirements

- Chrome or Chromium with Developer mode enabled
- Node.js `20.19+` or `22.12+`
- An authenticated WhatsApp Web session
- The trained model files, stored locally at:
  `assets/models/custom-macd-model/`

The model is intentionally excluded from source control.

## Build and load

```bash
npm ci
npm run build
```

Open `chrome://extensions`, enable **Developer mode**, choose **Load unpacked**,
and select `dist/`. Reload the extension and refresh WhatsApp Web.

The build bundles the extension, local model assets, ONNX Runtime WebAssembly,
and the PDF font. Runtime processing stays inside the browser.

## Code layout

- `content.js` – watches WhatsApp Web, finds incoming messages, shows warnings
- `banner.js` – the warning banner and bubble highlight
- `sw.js` – service worker: receives messages, queues them, saves incidents
- `classifier.js` – loads the on-device model and scores text
- `store.js` – local incident storage
- `popup.html` / `popup.js` / `pdf_exporter.js` – dashboard and exports

## Debugging

Inspect the service worker from `chrome://extensions`, or the WhatsApp Web page
console, using the normal Chrome DevTools. There is no built-in logging, so
message text is never written to the console.
