import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const DiabetesPage = () => {
  const [formData, setFormData] = useState({
    Age: "",
    Gender: "1",
    Polyuria: "0",
    Polydipsia: "0",
    SuddenWeightLost: "0",
    Weakness: "0",
    Polyphagia: "0",
    GenitalThrush: "0",
    VisualBlurring: "0",
    Itching: "0",
    Irritability: "0",
    DelayedHealing: "0",
    PartialParesis: "0",
    MuscleStiffness: "0",
    Alopecia: "0",
    Obesity: "0",
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
    fetch("http://localhost:3000/diabetes", {
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
            state: { response: 0, typeOfDisease: "Diabetes" },
          });
        } else {
          navigate("/response", {
            state: { response: 1, typeOfDisease: "Diabetes" },
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
        backgroundImage: `url('https://wallpapercave.com/wp/wp2616571.jpg')`,
        backgroundPosition: "center",
        backgroundSize: "cover",
        backgroundRepeat: "no-repeat",
        backgroundAttachment: "fixed",
      }}
    >
      <div className="container mx-[55%] w-[40%]">
        <h1 className="text-3xl font-bold py-3 text-center text-gray-700 mb-4">
          Diabetes Prediction{" "}
        </h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Input fields for diabetes prediction */}
          {/* Example: Age */}
          <div className="flex flex-col">
            <label htmlFor="Age" className="text-gray-900 mb-1 font-semibold">
              Age:
            </label>
            <input
              type="number"
              id="Age"
              name="Age"
              value={formData.Age}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Gender */}
          <div className="flex flex-col">
            <label
              htmlFor="Gender"
              className="text-gray-900 mb-1 font-semibold"
            >
              Gender:
            </label>
            <select
              id="Gender"
              name="Gender"
              value={formData.Gender}
              onChange={handleChange}
              className="border border-gray-300 rounded  bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Male</option>
              <option value="0">Female</option>
            </select>
          </div>

          {/* Example: Polyuria */}
          <div className="flex flex-col">
            <label
              htmlFor="Polyuria"
              className="text-gray-900 mb-1 font-semibold"
            >
              Polyuria:
            </label>
            <select
              id="Polyuria"
              name="Polyuria"
              value={formData.Polyuria}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Polydipsia */}
          <div className="flex flex-col">
            <label
              htmlFor="Polydipsia"
              className="text-gray-900 mb-1 font-semibold"
            >
              Polydipsia:
            </label>
            <select
              id="Polydipsia"
              name="Polydipsia"
              value={formData.Polydipsia}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Sudden Weight Loss */}
          <div className="flex flex-col">
            <label
              htmlFor="Sudden Weight Loss"
              className="text-gray-900 mb-1 font-semibold"
            >
              Sudden Weight Loss:
            </label>
            <select
              id="Sudden Weight Loss"
              name="Sudden Weight Loss"
              value={formData.SuddenWeightLoss}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Weakness */}
          <div className="flex flex-col">
            <label
              htmlFor="Weakness"
              className="text-gray-900 mb-1 font-semibold"
            >
              Weakness:
            </label>
            <select
              id="Weakness"
              name="Weakness"
              value={formData.Weakness}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Polyphagia */}
          <div className="flex flex-col">
            <label
              htmlFor="Polyphagia"
              className="text-gray-900 mb-1 font-semibold"
            >
              Polyphagia:
            </label>
            <select
              id="Polyphagia"
              name="Polyphagia"
              value={formData.Polyphagia}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Genital Thrush */}
          <div className="flex flex-col">
            <label
              htmlFor="Genital Thrush"
              className="text-gray-900 mb-1 font-semibold"
            >
              Genital Thrush:
            </label>
            <select
              id="Genital Thrush"
              name="Genital Thrush"
              value={formData.Genital_Thrush}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Visual Blurring */}
          <div className="flex flex-col">
            <label
              htmlFor="Visual Blurring"
              className="text-gray-900 mb-1 font-semibold"
            >
              Visual Blurring:
            </label>
            <select
              id="Visual Blurring"
              name="Visual Blurring"
              value={formData.Visual_Blurring}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Itching */}
          <div className="flex flex-col">
            <label
              htmlFor="Itching"
              className="text-gray-900 mb-1 font-semibold"
            >
              Itching:
            </label>
            <select
              id="Itching"
              name="Itching"
              value={formData.Itching}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Irritability */}
          <div className="flex flex-col">
            <label
              htmlFor="Irritability"
              className="text-gray-900 mb-1 font-semibold"
            >
              Irritability:
            </label>
            <select
              id="Irritability"
              name="Irritability"
              value={formData.Irritability}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Delayed Healing */}
          <div className="flex flex-col">
            <label
              htmlFor="Delayed Healing"
              className="text-gray-900 mb-1 font-semibold"
            >
              Delayed Healing:
            </label>
            <select
              id="Delayed Healing"
              name="Delayed Healing"
              value={formData.Delayed_Healing}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Partial Paresis */}
          <div className="flex flex-col">
            <label
              htmlFor="Partial Paresis"
              className="text-gray-900 mb-1 font-semibold"
            >
              Partial Paresis:
            </label>
            <select
              id="Partial Paresis"
              name="Partial Paresis"
              value={formData.Partial_Paresis}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Muscle Stiffness */}
          <div className="flex flex-col">
            <label
              htmlFor="Muscle Stiffness"
              className="text-gray-900 mb-1 font-semibold"
            >
              Muscle Stiffness:
            </label>
            <select
              id="Muscle Stiffness"
              name="Muscle Stiffness"
              value={formData.Muscle_Stiffness}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Alopecia */}
          <div className="flex flex-col">
            <label
              htmlFor="Alopecia"
              className="text-gray-900 mb-1 font-semibold"
            >
              Alopecia:
            </label>
            <select
              id="Alopecia"
              name="Alopecia"
              value={formData.Alopecia}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Other input fields for diabetes prediction */}
          {/* Example: Obesity */}
          <div className="flex flex-col">
            <label
              htmlFor="Obesity"
              className="text-gray-900 mb-1 font-semibold"
            >
              Obesity:
            </label>
            <select
              id="Obesity"
              name="Obesity"
              value={formData.Obesity}
              onChange={handleChange}
              className="border border-gray-300 rounded bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          {/* Submit button */}
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

export default DiabetesPage;
