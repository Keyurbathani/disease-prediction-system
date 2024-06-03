import React from "react";
import { useLocation } from "react-router-dom";

const ResponsePage = () => {
  const location = useLocation();
  const { response, typeOfDisease } = location.state;

  return (
    <div className="flex justify-center items-center h-screen bg-gray-200">
      <div className="bg-white p-8 rounded shadow-md">
        <h2 className="text-2xl font-bold mb-4">
          Prediction Result for {typeOfDisease} Disease
        </h2>
        <div className="text-lg">
          {response === 0 ? (
            <p className="text-green-600">
              No signs of {typeOfDisease} disease detected.
            </p>
          ) : (
            <p className="text-red-600">
              Warning: Indications of {typeOfDisease} disease found.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResponsePage;
