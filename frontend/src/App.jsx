// import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import HeartPage from "./pages/HeartPage";
import LiverPage from "./pages/LiverPage";
import KidneyPage from "./pages/KidneyPage";
import DiabetesPage from "./pages/DiabetesPage";
import HomePage from "./pages/HomePage";
import ResponsePage from "./pages/ResponsePage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/heart" element={<HeartPage />} />
        <Route path="/liver" element={<LiverPage />} />
        <Route path="/kidney" element={<KidneyPage />} />
        <Route path="/diabetes" element={<DiabetesPage />} />
        <Route path="/response" element={<ResponsePage />} />
        <Route path="/" element={<HomePage />} />
      </Routes>
    </Router>
  );
}

export default App;
