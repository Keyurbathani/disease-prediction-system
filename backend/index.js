const express = require('express');
const bodyParser = require('body-parser');
const heartRoutes = require('./routes/heart');
const kidneyRoutes = require('./routes/kidney');
const liverRoutes = require('./routes/liver');
const diabetesRoutes = require('./routes/diabetes');
// const authMiddleware = require('./middleware/authMiddleware');

const app = express();
const PORT = process.env.PORT || 3000;
const cors = require('cors');
app.use(cors(
  {
    origin: 'http://localhost:5173'
  }
));

// Middleware to parse incoming JSON data
app.use(bodyParser.json());

// Routes
// app.use('/heart', authMiddleware, heartRoutes);
// app.use('/kidney', authMiddleware, kidneyRoutes);
app.use('/heart', heartRoutes);
app.use('/kidney', kidneyRoutes);
app.use('/liver', liverRoutes);
app.use('/diabetes', diabetesRoutes);

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
