# Faceapi & Handpose integration

Conclusions:

1. Import faceapi first because it's a more abstract wrapper 2. Use this face-api unpkg repo and not the official one (compatibility errors) 3. WASM backend for Handpose is a must because WEBGL conflicts with handpose and bad things happen 4. Tinyfacedetector and not SSD faceapi, SSD doesn't work with Handpose 5. I understand why ML on the web isn't really a thing yet
