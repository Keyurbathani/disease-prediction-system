import React from "react";
import { BrowserRouter as Router, Route, NavLink } from "react-router-dom";
import HeartPage from "./HeartPage";
import LiverPage from "./LiverPage";
import KidneyPage from "./KidneyPage";
import DiabetesPage from "./DiabetesPage";

const NavigationHeader = () => {
  return (
    <nav className="flex justify-center w-full items-center bg-gray-900 p-4 border-t-0 border-b-[1px]">
      <NavLink
        to="/"
        exact
        activeClassName="text-white"
        className="text-gray-100  py-3 mx-6 rounded-3xl px-1 hover:text-white transition-colors duration-300 font-semibold text-lg"
      >
        Home
      </NavLink>
      <NavLink
        to="/heart"
        activeClassName="text-white"
        className="text-gray-100  py-3 mx-6 rounded-xl px-1 hover:text-white transition-colors duration-300 font-semibold text-lg"
      >
        Heart
      </NavLink>
      <NavLink
        to="/liver"
        activeClassName="text-white"
        className="text-gray-100  py-3 mx-6 rounded-xl px-1 hover:text-white transition-colors duration-300 font-semibold text-lg"
      >
        Liver
      </NavLink>
      <NavLink
        to="/kidney"
        activeClassName="text-white"
        className="text-gray-100 py-3 mx-6 rounded-xl px-1 hover:text-white transition-colors duration-300 font-semibold text-lg"
      >
        Kidney
      </NavLink>
      <NavLink
        to="/diabetes"
        activeClassName="text-white"
        className="text-gray-100  py-3 mx-6 rounded-xl px-1 hover:text-white transition-colors duration-300 font-semibold text-lg"
      >
        Diabetes
      </NavLink>
    </nav>
  );
};

const HomePage = () => {
  return (
    <div className="w-full flex flex-col mx-auto bg-gray-800">
      <NavigationHeader />

      <h1 className=" pb-5 pt-5 text-6xl text-white text-center font-bold border-b-[1px]">
        Disease Prediction
      </h1>
      <div className="bg-gray-800 p-8 flex items-center border-b-[1px]">
        <div className="w-1/2">
          <img
            src="https://wallpapers.com/images/hd/black-and-white-human-heart-hd-medical-we31xzxdohwrp24b.jpg"
            alt="Your Photo"
            className="w-[80%] rounded-lg mx-auto shadow-[0_20px_25px_-5px_rgba(255,255,255,0.1)]"
          />
        </div>

        <div className="w-1/2 mx-auto px-3">
          <h2 className="text-4xl text-gray-100 font-bold mb-4">
            Heart Disease Prediction
          </h2>
          <div className="text-gray-300 text-lg text-justify">
            <ul className="list-disc">
              <li className="mb-4">
                Heart disease encompasses various conditions affecting the
                heart's function and structure, often linked to risk factors
                like high blood pressure and cholesterol levels. Early detection
                and lifestyle modifications are pivotal in mitigating its impact
                and preventing complications.
              </li>
            </ul>
          </div>
          <div className="">
            <a
              href="heart"
              class="inline-flex items-center px-3 py-2 text-sm font-medium text-center text-white bg-blue-700 rounded-lg hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
            >
              Predict Heart Disease
              <svg
                class="rtl:rotate-180 w-3.5 h-3.5 ms-2"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 14 10"
              >
                <path
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M1 5h12m0 0L9 1m4 4L9 9"
                />
              </svg>
            </a>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 p-8 flex items-center border-b-[1px]">
        <div className="w-1/2 mx-auto px-3">
          <h2 className="text-4xl  text-gray-100 font-bold mb-4">
            Liver Disease Prediction
          </h2>

          <div className="text-gray-300 text-lg text-justify">
            <ul className="list-disc">
              <li className="mb-4">
                Liver disease encompasses various conditions affecting liver
                function such as hepatitis, fatty liver disease, and cirrhosis.
                Risk factors include excessive alcohol intake, viral infections,
                obesity, and certain medications. These conditions pose a
                spectrum of challenges for the liver's health.
              </li>
            </ul>
          </div>
          <div className="">
            <a
              href="liver"
              class="inline-flex items-center px-3 py-2 text-sm font-medium text-center text-white bg-blue-700 rounded-lg hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
            >
              Predict Liver Disease
              <svg
                class="rtl:rotate-180 w-3.5 h-3.5 ms-2"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 14 10"
              >
                <path
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M1 5h12m0 0L9 1m4 4L9 9"
                />
              </svg>
            </a>
          </div>
        </div>

        <div className="w-1/2">
          <img
            src="https://www.organtransplantpune.com/wp-content/uploads/2020/11/iStock-970417366-liver-story.jpg"
            alt="Your Photo"
            className="w-[80%]  mx-auto rounded-lg shadow-[0_20px_25px_-5px_rgba(255,255,255,0.1)]"
          />
        </div>
      </div>

      <div className="bg-gray-800 p-8 flex items-center border-b-[1px]">
        <div className="w-1/2">
          <img
            src="https://pharmaphorum.com/sites/default/files/styles/x_large/public/2023-03/kidney-disease.jpg?itok=biuptZFz"
            alt="Your Photo"
            className="w-[80%] rounded-lg mx-auto shadow-[0_20px_25px_-5px_rgba(255,255,255,0.1)]"
          />
        </div>

        <div className="w-1/2 mx-auto px-3">
          <h2 className="text-4xl text-gray-100 font-bold mb-4">
            Kidney Disease Prediction
          </h2>
          <div className="text-gray-300 text-lg text-justify">
            <ul className="list-disc">
              <li className="mb-4">
                Kidney disease encompasses chronic kidney disease (CKD), kidney
                stones, and kidney infections. Risk factors include diabetes,
                high blood pressure, family history, and certain medications.
                These conditions pose significant challenges to kidney health.
              </li>
            </ul>
          </div>
          <div className="">
            <a
              href="kidney"
              class="inline-flex items-center px-3 py-2 text-sm font-medium text-center text-white bg-blue-700 rounded-lg hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
            >
              Kidney Liver Disease
              <svg
                class="rtl:rotate-180 w-3.5 h-3.5 ms-2"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 14 10"
              >
                <path
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M1 5h12m0 0L9 1m4 4L9 9"
                />
              </svg>
            </a>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 p-8 flex items-center border-b-[1px]">
        <div className="w-1/2 mx-auto px-3">
          <h2 className="text-4xl  text-gray-100 font-bold mb-4">
            Diabetes Disease Prediction
          </h2>

          <div className="text-gray-300 text-lg text-justify">
            <ul className="list-disc">
              <li className="mb-4">
                Diabetes is a metabolic disorder characterized by high blood
                sugar levels, either due to insufficient insulin production or
                ineffective use of insulin by the body. Risk factors include
                obesity, unhealthy diet, physical inactivity, and genetic
                predisposition.
              </li>
            </ul>
          </div>
          <div className="">
            <a
              href="diabetes"
              class="inline-flex items-center px-3 py-2 text-sm font-medium text-center text-white bg-blue-700 rounded-lg hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
            >
              Predict Diabetes Disease
              <svg
                class="rtl:rotate-180 w-3.5 h-3.5 ms-2"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 14 10"
              >
                <path
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M1 5h12m0 0L9 1m4 4L9 9"
                />
              </svg>
            </a>
          </div>
        </div>

        <div className="w-1/2">
          <img
            src="https://wallpapercave.com/wp/wp2616571.jpg"
            alt="Your Photo"
            className="w-[80%]  mx-auto rounded-lg shadow-[0_20px_25px_-5px_rgba(255,255,255,0.1)]"
          />
        </div>
      </div>
    </div>
  );
};

export default HomePage;
