WHY ONNX ??
- goto way for transformer.js models 
- transformers.js provided a wrapper pipeline that is a wrapperon top of it 

ONNX Runtime Web- browser ke andar model chalata hai. Python ya server nahi chahiye

ultimatel t ONNX runtime web provides an optimized ONNX runtime .

The Open Neural Network Exchange (ONNX) is open standard for representing machine learning models

RUN ON BOTH CPU AND GPU ..


Training PyTorch mein hui. Browser PyTorch model nahi chala sakta

the ONNX runtime ka CPu logic is compiled into web assembly which is run on the browser .

WebAssembly
WebAssembly (Wasm) is a low-level assembly-like language that brings near-native performance to the web. It serves as a compilation target for languages like C/C++, C#, and Rust, allowing high-performance code to run directly in the browser.

WebAssembly is designed to complement and run alongside JavaScript. Using the WebAssembly JavaScript APIs, you can load WebAssembly modules into a JavaScript app and share functionality between the two. This allows you to take advantage of WebAssembly's performance and power and JavaScript's expressiveness and flexibility in the same app — without having to write low-level WebAssembly code yourself.



THE JS BROWSER OF THE THREAD IS running the content.js 
the background is runnign sw.js which is also running the model waala onnx runtime inferencing logic 
they both communicate via chrome.runtime.sendMessage



