const express = require("express");
const axios = require("axios");
const app = express();

app.use(express.json());

app.post("/predict", async (req, res) => {
  const response = await axios.post("http://localhost:5000/predict", req.body);
  res.json(response.data);
});

app.listen(3000, () => console.log("Node server running"));
