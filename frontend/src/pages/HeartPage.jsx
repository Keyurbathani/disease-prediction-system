import React from "react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

const HeartPage = () => {
  const [formData, setFormData] = useState({
    age: "",
    sex: "1",
    chestPain: "0",
    restingBp: "",
    cholesterol: "",
    fastingBloodSugar: "0",
    restingEcg: "0",
    maxHeartRate: "",
    exerciseAngina: "0",
    oldpeak: "",
    stSlope: "0",
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
    fetch("http://localhost:3000/heart", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    })
      .then((response) => response.json())
      .then((data) => {
        setResponse(JSON.stringify(data));
        console.log(data.predictions[0]);
        if (data.predictions[0] === 0) {
          navigate("/response", {
            state: { response: 0, typeOfDisease: "Heart" },
          });
        } else {
          navigate("/response", {
            state: { response: 1, typeOfDisease: "Heart" },
          });
        }
        // history.push("/response", { response: data, typeOfDisease: "Heart" });
      })
      .catch((error) => {
        setResponse("An error occurred: " + error);
      });
  };

  return (
    <div
      style={{
        backgroundImage: `url('https://wallpapercave.com/wp/wp11352657.jpg')`,
        backgroundPosition: "center",
        backgroundSize: "cover",
        backgroundRepeat: "no-repeat",
        backgroundAttachment: "fixed",
      }}
    >
      <div className="container mx-[50%] w-[40%]">
        <h1 className="text-3xl font-bold py-3 text-center text-white shadow-md mb-4">
          Heart Disease Prediction{" "}
        </h1>
        <form onSubmit={handleSubmit} className="space-y-4">
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

          <div className="flex flex-col">
            <label htmlFor="sex" className="text-white mb-1 font-semibold">
              Sex:
            </label>
            <select
              id="sex"
              name="sex"
              value={formData.sex}
              onChange={handleChange}
              className="border border-gray-300 rounded  bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="1">Male</option>
              <option value="0">Female</option>
            </select>
          </div>

          <div className="flex flex-col">
            <label
              htmlFor="chestPain"
              className="text-white mb-1 font-semibold"
            >
              Chest Pain Type:
            </label>
            <select
              id="chestPain"
              name="chestPain"
              value={formData.chestPain}
              onChange={handleChange}
              className="border border-gray-300  bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="0">Typical angina</option>
              <option value="1">Atypical angina</option>
              <option value="2">Non-anginal pain</option>
              <option value="3">Asymptomatic</option>
            </select>
          </div>

          <div className="flex flex-col">
            <label
              htmlFor="restingBp"
              className="text-white mb-1 font-semibold"
            >
              Resting Blood Pressure:
            </label>
            <input
              type="number"
              id="restingBp"
              name="restingBp"
              value={formData.restingBp}
              onChange={handleChange}
              className="border border-gray-300  bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          <div className="flex flex-col">
            <label
              htmlFor="cholesterol"
              className="text-white mb-1 font-semibold"
            >
              Cholesterol:
            </label>
            <input
              type="number"
              id="cholesterol"
              name="cholesterol"
              value={formData.cholesterol}
              onChange={handleChange}
              className="border border-gray-300  bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          <div className="flex flex-col">
            <label
              htmlFor="fastingBloodSugar"
              className="text-white mb-1 font-semibold"
            >
              Fasting Blood Sugar:
            </label>
            <select
              id="fastingBloodSugar"
              name="fastingBloodSugar"
              value={formData.fastingBloodSugar}
              onChange={handleChange}
              className="border border-gray-300  bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="0">False (&lt;= 120 mg/dl)</option>
              <option value="1">True (&gt; 120 mg/dl)</option>
            </select>
          </div>

          <div className="flex flex-col">
            <label
              htmlFor="restingEcg"
              className="text-white mb-1 font-semibold"
            >
              Resting ECG:
            </label>
            <select
              id="restingEcg"
              name="restingEcg"
              value={formData.restingEcg}
              onChange={handleChange}
              className="border border-gray-300  bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="0">Normal</option>
              <option value="1">ST-T wave abnormality</option>
              <option value="2">Left ventricular hypertrophy</option>
            </select>
          </div>

          <div className="flex flex-col">
            <label
              htmlFor="maxHeartRate"
              className="text-white mb-1 font-semibold"
            >
              Maximum Heart Rate:
            </label>
            <input
              type="number"
              id="maxHeartRate"
              name="maxHeartRate"
              value={formData.maxHeartRate}
              onChange={handleChange}
              className="border border-gray-300  bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          <div className="flex flex-col">
            <label
              htmlFor="exerciseAngina"
              className="text-white mb-1 font-semibold"
            >
              Exercise Induced Angina:
            </label>
            <select
              id="exerciseAngina"
              name="exerciseAngina"
              value={formData.exerciseAngina}
              onChange={handleChange}
              className="border border-gray-300  bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </div>

          <div className="flex flex-col">
            <label htmlFor="oldpeak" className="text-white mb-1 font-semibold">
              ST Depression:
            </label>
            <input
              type="number"
              id="oldpeak"
              name="oldpeak"
              value={formData.oldpeak}
              onChange={handleChange}
              className="border border-gray-300  bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="stSlope" className="text-white mb-1 font-semibold">
              ST Slope:
            </label>
            <select
              id="stSlope"
              name="stSlope"
              value={formData.stSlope}
              onChange={handleChange}
              className="border border-gray-300  bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            >
              <option value="0">Upsloping</option>
              <option value="1">Flat</option>
              <option value="2">Downsloping</option>
            </select>
          </div>
          <div className="flex flex-col items-center pb-5">
            <button
              type="submit"
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded "
            >
              Submit
            </button>
          </div>
        </form>
        {/* <div>{response}</div> */}
      </div>
    </div>
  );
};

export default HeartPage;
