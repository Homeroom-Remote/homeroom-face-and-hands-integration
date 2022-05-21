// configuration options
const modelPath = "./models/"; // path to model folder that will be loaded using http
// const modelPath = 'https://vladmandic.github.io/face-api/model/'; // path to model folder that will be loaded using http
const imgSize = 800; // maximum image size in pixels
const minScore = 0.3; // minimum score
const maxResults = 10; // maximum number of results to return
const samples = [
  "sample1.jpg",
  "sample2.jpg",
  "sample3.jpg",
  "sample4.jpg",
  "sample5.jpg",
  "sample6.jpg",
]; // sample images to be loaded using http

// helper function to pretty-print json object to string
const str = (json) =>
  json
    ? JSON.stringify(json)
        .replace(/{|}|"|\[|\]/g, "")
        .replace(/,/g, ", ")
    : "";

// helper function to print strings to html document as a log
function log(...txt) {
  // eslint-disable-next-line no-console
  console.log(...txt);
}

// loads image and draws it on resized canvas so we alwys have correct image size regardless of source
async function image(url) {
  return new Promise((resolve) => {
    const img = new Image();
    // wait until image is actually loaded
    img.addEventListener("load", () => {
      // resize image so larger axis is not bigger than limit
      const ratio = (1.0 * img.height) / img.width;
      img.width = ratio <= 1 ? imgSize : (1.0 * imgSize) / ratio;
      img.height = ratio >= 1 ? imgSize : 1.0 * imgSize * ratio;
      // create canvas and draw loaded image
      const canvas = document.createElement("canvas");
      canvas.height = img.height;
      canvas.width = img.width;
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.drawImage(img, 0, 0, img.width, img.height);
      // return generated canvas to be used by tfjs during detection
      resolve(canvas);
    });
    // load image
    img.src = url;
  });
}

const vid = document.getElementById("vid");
if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then(function (stream) {
      vid.srcObject = stream;
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
}

async function main() {
  // initialize tfjs
  log("FaceAPI Test");

  const params = new URLSearchParams(location.search);
  if (params.has("backend")) {
    const backend = params.get("backend");
    await faceapi.tf.setWasmPaths(
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@3.17.0/dist/"
    );
    log(`Chosen backend: ${backend}`);
    await faceapi.tf.setBackend(backend);
  } else {
    // default is webgl backend
    await faceapi.tf.setBackend("webgl");
  }

  await faceapi.tf.enableProdMode();
  await faceapi.tf.ENV.set("DEBUG", false);
  await faceapi.tf.ready();

  // check version
  log(
    `Version: FaceAPI ${str(
      faceapi?.version || "(not loaded)"
    )} TensorFlow/JS ${str(
      faceapi?.tf?.version_core || "(not loaded)"
    )} Backend: ${str(faceapi?.tf?.getBackend() || "(not loaded)")}`
  );
  log(
    `Flags: ${JSON.stringify(faceapi?.tf?.ENV.flags || { tf: "not loaded" })}`
  );

  // load face-api models
  log("Loading FaceAPI models");
  await faceapi.nets.tinyFaceDetector.load(modelPath);
  await faceapi.nets.ssdMobilenetv1.load(modelPath);
  await faceapi.nets.faceExpressionNet.load(modelPath);
  const optionsTinyFace = new faceapi.TinyFaceDetectorOptions({
    inputSize: imgSize,
    scoreThreshold: minScore,
  });

  // check tf engine state
  const engine = await faceapi.tf.engine();
  log(`TF Engine State: ${str(engine.state)}`);

  const model = await handpose.load();
  const GestureEstimator = new fp.GestureEstimator([
    fp.Gestures.VictoryGesture,
    fp.Gestures.ThumbsUpGesture,
  ]);
  vid.addEventListener("loadeddata", () => {
    setInterval(async () => {
      try {
        const hand = await model.estimateHands(vid, true);

        if (hand.length > 0) {
          const gesture = await GestureEstimator.estimate(
            hand[0].landmarks,
            8.5
          );

          if (gesture.gestures?.length > 0) {
            const confidence = gesture.gestures.map(
              (prediction) => prediction.score
            );
            const maxConfidence = confidence.indexOf(
              Math.max.apply(null, confidence)
            );

            log(gesture.gestures[maxConfidence].name);
          }
        }

        // actual model execution
        const dataTinyYolo = await faceapi
          // @ts-ignore
          .detectAllFaces(vid, optionsTinyFace)
          .withFaceExpressions();
        // log results to screen
        log("TinyFace:", dataTinyYolo[0].expressions);
        // actual model execution
      } catch (err) {
        log(`Error during processing ${str(err)}`);
        // eslint-disable-next-line no-console
        console.error(err);
      }
    }, 500);
  });
}

// start processing as soon as page is loaded
window.onload = main;
