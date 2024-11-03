const functions = require("firebase-functions");
const express = require("express");
const { spawn } = require("child_process");

const app = express();

app.post("/predict", (req, res) => {
  const subprocess = spawn("python", ["../main.py"]);

  subprocess.stdout.on("data", (data) => {
    res.send(data.toString());
  });

  subprocess.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  subprocess.on("close", (code) => {
    console.log(`child process exited with code ${code}`);
  });
});

exports.api = functions.https.onRequest(app);
