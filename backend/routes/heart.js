const express = require('express');
const { spawn } = require('child_process');
const path = require('path');

const router = express.Router();

router.post('/', (req, res) => {
  // Extract input data from request body
  const inputData = req.body;
  console.log("Data received for heart route:", inputData);

  // Prepare input data as a JSON string
  const jsonData = JSON.stringify(inputData);

  // Define Python script file path
  const pythonScript = path.join(__dirname, '../scripts/heartPredictions.py');

  // Spawn a new Python process
  const pythonProcess = spawn('python', [pythonScript, jsonData]);

  // Handle stdout data from Python script
  pythonProcess.stdout.on('data', (data) => {
    console.log("Python script output:", data.toString());
    try {
      const predictions = JSON.parse(data);
      console.log("Predictions:", predictions);
      res.json({ predictions });
    } catch (error) {
      console.error("Error parsing JSON:", error.message);
      res.status(500).json({ error: 'An error occurred while parsing predictions.' });
    }
  });

  // Handle stderr data from Python script
  // pythonProcess.stderr.on('data', (data) => {
  //   console.error("Python script error:", data.toString());
  //   res.status(500).json({ error: 'An error occurred while executing the Python script.' });
  // });

  // Handle Python script process exit
  pythonProcess.on('exit', (code) => {
    console.log(`Python script process exited with code ${code}`);
  });
});

module.exports = router;
