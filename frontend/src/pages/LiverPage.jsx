import  { useState } from "react";
import { useNavigate } from "react-router-dom";

const LiverPage = () => {
  const [formData, setFormData] = useState({
    age: "",
    gender: "0",
    tb: "",
    db: "",
    alkphos: "",
    sgpt: "",
    sgot: "",
    tp: "",
    alb: "",
    ag: "",
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
    fetch("http://localhost:3000/liver", {
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
            state: { response: 0, typeOfDisease: "Liver" },
          });
        } else {
          navigate("/response", {
            state: { response: 1, typeOfDisease: "Liver" },
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
        backgroundImage: `url('https://www.organtransplantpune.com/wp-content/uploads/2020/11/iStock-970417366-liver-story.jpg')`,
        backgroundPosition: "center",
        backgroundSize: "cover",
        backgroundRepeat: "no-repeat",
        backgroundAttachment: "fixed",
      }}
    >
      <div className="container mx-[55%] w-[40%]">
        <h1 className="text-3xl font-bold py-3 text-center text-white shadow-md mb-4">
          Liver Disease Prediction{" "}
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
            <label htmlFor="gender" className="text-white mb-1 font-semibold">
              Gender:
            </label>
            <select
              id="gender"
              name="gender"
              value={formData.gender}
              onChange={handleChange}
              className="border border-gray-300 rounded  bg-gray-200 shadow-lg px-3 py-2"
              required
            >
              <option value="0">Male</option>
              <option value="1">Female</option>
            </select>
          </div>

          <div className="flex flex-col">
            <label htmlFor="tb" className="text-white mb-1 font-semibold">
              Total Bilirubin (tb):
            </label>
            <input
              type="number"
              id="tb"
              name="tb"
              value={formData.tb}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="db" className="text-white mb-1 font-semibold">
              Direct Bilirubin (db):
            </label>
            <input
              type="number"
              id="db"
              name="db"
              value={formData.db}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="alkphos" className="text-white mb-1 font-semibold">
              Alkaline Phosphotase (alkphos):
            </label>
            <input
              type="number"
              id="alkphos"
              name="alkphos"
              value={formData.alkphos}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="sgpt" className="text-white mb-1 font-semibold">
              Alamine Aminotransferase (sgpt):
            </label>
            <input
              type="number"
              id="sgpt"
              name="sgpt"
              value={formData.sgpt}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="sgot" className="text-white mb-1 font-semibold">
              Aspartate Aminotransferase (sgot):
            </label>
            <input
              type="number"
              id="sgot"
              name="sgot"
              value={formData.sgot}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="tp" className="text-white mb-1 font-semibold">
              Total Proteins (tp):
            </label>
            <input
              type="number"
              id="tp"
              name="tp"
              value={formData.tp}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="alb" className="text-white mb-1 font-semibold">
              Albumin (alb):
            </label>
            <input
              type="number"
              id="alb"
              name="alb"
              value={formData.alb}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
          </div>

          <div className="flex flex-col">
            <label htmlFor="ag" className="text-white mb-1 font-semibold">
              Albumin/Globulin Ratio (a/g):
            </label>
            <input
              type="number"
              id="ag"
              name="ag"
              value={formData.ag}
              onChange={handleChange}
              className="border border-gray-300 bg-gray-200 shadow-lg rounded px-3 py-2"
              required
            />
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
        <div>{response}</div>
      </div>
    </div>
  );
};

export default LiverPage;
