import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const KidneyPage = () => {
  const [formData, setFormData] = useState({
    age: "",
    blood_pressure: "",
    specific_gravity: "",
    albumin: "",
    sugar: "",
    red_blood_cells: "normal",
    pus_cell: "notpresent",
    pus_cell_clumps: "notpresent",
    bacteria: "notpresent",
    blood_glucose_random: "",
    blood_urea: "",
    serum_creatinine: "",
    sodium: "",
    potassium: "",
    haemoglobin: "",
    packed_cell_volume: "",
    white_blood_cell_count: "",
    red_blood_cell_count: "",
    hypertension: "no",
    diabetes_mellitus: "no",
    coronary_artery_disease: "no",
    appetite: "good",
    pedal_edema: "no",
    anemia: "no",
  });
  const [response, setResponse] = useState("");
  const navigate = useNavigate();

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    fetch("http://localhost:3000/kidney", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    })
      .then((response) => response.json())
      .then((data) => {
        setResponse(JSON.stringify(data));
        if (data.predictions[0] === 0) {
          navigate("/response", {
            state: { response: 0, typeOfDisease: "Kidney" },
          });
        } else {
          navigate("/response", {
            state: { response: 1, typeOfDisease: "Kidney" },
          });
        }
      })
      .catch((error) => {
        setResponse("An error occurred: " + error);
      });
  };

  return (
    <div
      style={{
        backgroundImage: `url('https://www.paho.org/sites/default/files/styles/max_1500x1500/public/hero/2019-10/kidney.jpg?itok=VbU-yU-Z')`,
        backgroundPosition: "center",
        backgroundSize: "cover",
        backgroundRepeat: "no-repeat",
        backgroundAttachment: "fixed",
      }}
    >
      <div className="container mx-[55%] w-[40%]">
        <h1 className="text-3xl font-bold py-3 text-center text-white shadow-md mb-4">
          Kidney Disease Prediction{" "}
        </h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Age */}
          <div className="flex flex-col">
            <label htmlFor="age" className="text-white mb-1 font-semibold">
              Age:
            </label>
            <input
              type="number"
              id="age"
              name="age"
              value={formData.age}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Blood Pressure */}
          <div className="flex flex-col">
            <label
              htmlFor="blood_pressure"
              className="text-white mb-1 font-semibold"
            >
              Blood Pressure:
            </label>
            <input
              type="number"
              id="blood_pressure"
              name="blood_pressure"
              value={formData.blood_pressure}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Specific Gravity */}
          <div className="flex flex-col">
            <label
              htmlFor="specific_gravity"
              className="text-white mb-1 font-semibold"
            >
              Specific Gravity:
            </label>
            <input
              type="number"
              id="specific_gravity"
              name="specific_gravity"
              value={formData.specific_gravity}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Albumin */}
          <div className="flex flex-col">
            <label htmlFor="albumin" className="text-white mb-1 font-semibold">
              Albumin:
            </label>
            <select
              id="albumin"
              name="albumin"
              value={formData.albumin}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
            </select>
          </div>

          {/* Sugar */}
          <div className="flex flex-col">
            <label htmlFor="sugar" className="text-white mb-1 font-semibold">
              Sugar:
            </label>
            <select
              id="sugar"
              name="sugar"
              value={formData.sugar}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
            </select>
          </div>

          {/* Red Blood Cells */}
          <div className="flex flex-col">
            <label
              htmlFor="red_blood_cells"
              className="text-white mb-1 font-semibold"
            >
              Red Blood Cells:
            </label>
            <select
              id="red_blood_cells"
              name="red_blood_cells"
              value={formData.red_blood_cells}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="normal">Normal</option>
              <option value="abnormal">Abnormal</option>
            </select>
          </div>

          {/* Pus Cells */}
          <div className="flex flex-col">
            <label htmlFor="pus_cell" className="text-white mb-1 font-semibold">
              Pus Cells:
            </label>
            <select
              id="pus_cell"
              name="pus_cell"
              value={formData.pus_cell}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="notpresent">Not Present</option>
              <option value="present">Present</option>
            </select>
          </div>

          {/* Pus Cell Clumps */}
          <div className="flex flex-col">
            <label
              htmlFor="pus_cell_clumps"
              className="text-white mb-1 font-semibold"
            >
              Pus Cell Clumps:
            </label>
            <select
              id="pus_cell_clumps"
              name="pus_cell_clumps"
              value={formData.pus_cell_clumps}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="notpresent">Not Present</option>
              <option value="present">Present</option>
            </select>
          </div>

          {/* Bacteria */}
          <div className="flex flex-col">
            <label htmlFor="bacteria" className="text-white mb-1 font-semibold">
              Bacteria:
            </label>
            <select
              id="bacteria"
              name="bacteria"
              value={formData.bacteria}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="notpresent">Not Present</option>
              <option value="present">Present</option>
            </select>
          </div>

          {/* Blood Glucose Random */}
          <div className="flex flex-col">
            <label
              htmlFor="blood_glucose_random"
              className="text-white mb-1 font-semibold"
            >
              Blood Glucose Random:
            </label>
            <input
              type="number"
              id="blood_glucose_random"
              name="blood_glucose_random"
              value={formData.blood_glucose_random}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Blood Urea */}
          <div className="flex flex-col">
            <label
              htmlFor="blood_urea"
              className="text-white mb-1 font-semibold"
            >
              Blood Urea:
            </label>
            <input
              type="number"
              id="blood_urea"
              name="blood_urea"
              value={formData.blood_urea}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Serum Creatinine */}
          <div className="flex flex-col">
            <label
              htmlFor="serum_creatinine"
              className="text-white mb-1 font-semibold"
            >
              Serum Creatinine:
            </label>
            <input
              type="number"
              id="serum_creatinine"
              name="serum_creatinine"
              value={formData.serum_creatinine}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Sodium */}
          <div className="flex flex-col">
            <label htmlFor="sodium" className="text-white mb-1 font-semibold">
              Sodium:
            </label>
            <input
              type="number"
              id="sodium"
              name="sodium"
              value={formData.sodium}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Potassium */}
          <div className="flex flex-col">
            <label
              htmlFor="potassium"
              className="text-white mb-1 font-semibold"
            >
              Potassium:
            </label>
            <input
              type="number"
              id="potassium"
              name="potassium"
              value={formData.potassium}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Hemoglobin */}
          <div className="flex flex-col">
            <label
              htmlFor="haemoglobin"
              className="text-white mb-1 font-semibold"
            >
              Hemoglobin:
            </label>
            <input
              type="number"
              id="haemoglobin"
              name="haemoglobin"
              value={formData.haemoglobin}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Packed Cell Volume */}
          <div className="flex flex-col">
            <label
              htmlFor="packed_cell_volume"
              className="text-white mb-1 font-semibold"
            >
              Packed Cell Volume:
            </label>
            <input
              type="number"
              id="packed_cell_volume"
              name="packed_cell_volume"
              value={formData.packed_cell_volume}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* White Blood Cell Count */}
          <div className="flex flex-col">
            <label
              htmlFor="white_blood_cell_count"
              className="text-white mb-1 font-semibold"
            >
              White Blood Cell Count:
            </label>
            <input
              type="number"
              id="white_blood_cell_count"
              name="white_blood_cell_count"
              value={formData.white_blood_cell_count}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Red Blood Cell Count */}
          <div className="flex flex-col">
            <label
              htmlFor="red_blood_cell_count"
              className="text-white mb-1 font-semibold"
            >
              Red Blood Cell Count:
            </label>
            <input
              type="number"
              id="red_blood_cell_count"
              name="red_blood_cell_count"
              value={formData.red_blood_cell_count}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Hypertension */}
          <div className="flex flex-col">
            <label
              htmlFor="hypertension"
              className="text-white mb-1 font-semibold"
            >
              Hypertension:
            </label>
            <select
              id="hypertension"
              name="hypertension"
              value={formData.hypertension}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="no">No</option>
              <option value="yes">Yes</option>
            </select>
          </div>

          {/* Diabetes Mellitus */}
          <div className="flex flex-col">
            <label
              htmlFor="diabetes_mellitus"
              className="text-white mb-1 font-semibold"
            >
              Diabetes Mellitus:
            </label>
            <select
              id="diabetes_mellitus"
              name="diabetes_mellitus"
              value={formData.diabetes_mellitus}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="no">No</option>
              <option value="yes">Yes</option>
            </select>
          </div>

          {/* Coronary Artery Disease */}
          <div className="flex flex-col">
            <label
              htmlFor="coronary_artery_disease"
              className="text-white mb-1 font-semibold"
            >
              Coronary Artery Disease:
            </label>
            <select
              id="coronary_artery_disease"
              name="coronary_artery_disease"
              value={formData.coronary_artery_disease}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="no">No</option>
              <option value="yes">Yes</option>
            </select>
          </div>

          {/* Appetite */}
          <div className="flex flex-col">
            <label htmlFor="appetite" className="text-white mb-1 font-semibold">
              Appetite:
            </label>
            <select
              id="appetite"
              name="appetite"
              value={formData.appetite}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="good">Good</option>
              <option value="poor">Poor</option>
            </select>
          </div>

          {/* Pedal Edema */}
          <div className="flex flex-col">
            <label
              htmlFor="pedal_edema"
              className="text-white mb-1 font-semibold"
            >
              Pedal Edema:
            </label>
            <select
              id="pedal_edema"
              name="pedal_edema"
              value={formData.pedal_edema}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="no">No</option>
              <option value="yes">Yes</option>
            </select>
          </div>

          {/* Anemia */}
          <div className="flex flex-col">
            <label htmlFor="anemia" className="text-white mb-1 font-semibold">
              Anemia:
            </label>
            <select
              id="anemia"
              name="anemia"
              value={formData.anemia}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="no">No</option>
              <option value="yes">Yes</option>
            </select>
          </div>

          <div className="flex flex-col items-center pb-5">
            <button
              type="submit"
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
            >
              Submit
            </button>
          </div>
        </form>
        <div>{response}</div>
      </div>
    </div>
  );
};

export default KidneyPage;
