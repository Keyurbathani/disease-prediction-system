import React from "react";
import { Link } from "react-router-dom";

function HomePage() {
  return (
    <div className="bg-slate-950">
      <div className="container mx-auto pb-9">
        <h1 className="text-4xl font-bold mb-6 text-center pt-5 text-white">
          Welcome to Disease Prediction
        </h1>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div class="max-w-sm bg-white border border-gray-200 rounded-lg shadow dark:bg-gray-800 dark:border-gray-700">
            <a href="heart">
              <img
                class="rounded-t-lg"
                src="https://wallpapers.com/images/hd/black-and-white-human-heart-hd-medical-we31xzxdohwrp24b.jpg"
                alt=""
              />
            </a>
            <div class="p-5 flex flex-col justify-between">
              <div>
                <a href="heart">
                  <h5 class="mb-2 text-2xl font-bold tracking-tight text-center text-gray-900 dark:text-white">
                    Heart Disease Prediction
                  </h5>
                </a>
                <p class="mb-3 font-normal text-gray-700 dark:text-gray-400 text-center">
                  Heart disease encompasses various conditions affecting the
                  heart's function and structure, often linked to risk factors
                  like high blood pressure and cholesterol levels. Early
                  detection and lifestyle modifications are pivotal in
                  mitigating its impact and preventing complications.
                </p>
              </div>
              <div className="mt-auto">
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

          <div class="max-w-sm bg-white border border-gray-200 rounded-lg shadow dark:bg-gray-800 dark:border-gray-700">
            <a href="liver">
              <img
                class="rounded-t-lg"
                src="https://www.organtransplantpune.com/wp-content/uploads/2020/11/iStock-970417366-liver-story.jpg"
                alt=""
              />
            </a>
            <div class="p-5">
              <a href="liver">
                <h5 class="mb-2 text-2xl font-bold tracking-tight text-center text-gray-900 dark:text-white">
                  Liver Disease Prediction
                </h5>
              </a>
              <p className="mb-3 font-normal text-gray-700 dark:text-gray-400 text-center ">
                Liver disease encompasses various conditions affecting liver
                function such as hepatitis, fatty liver disease, and cirrhosis.
                Risk factors include excessive alcohol intake, viral infections,
                obesity, and certain medications. These conditions pose a
                spectrum of challenges for the liver's health.
              </p>
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

          <div class="max-w-sm bg-white border border-gray-200 rounded-lg shadow dark:bg-gray-800 dark:border-gray-700">
            <a href="kidney">
              <img
                class="rounded-t-lg"
                src="https://pharmaphorum.com/sites/default/files/styles/x_large/public/2023-03/kidney-disease.jpg?itok=biuptZFz"
                alt=""
              />
            </a>
            <div class="p-5">
              <a href="kidney">
                <h5 class="mb-2 text-2xl font-bold tracking-tight text-center text-gray-900 dark:text-white">
                   predict Kidney Disease
                </h5>
              </a>
              <p className="mb-3 font-normal text-gray-700 dark:text-gray-400 text-center">
                Kidney disease encompasses chronic kidney disease (CKD), kidney
                stones, and kidney infections. Risk factors include diabetes,
                high blood pressure, family history, and certain medications.
                These conditions pose significant challenges to kidney health.
              </p>
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

          <div class="max-w-sm bg-white border border-gray-200 rounded-lg shadow dark:bg-gray-800 dark:border-gray-700">
            <a href="diabetes">
              <img
                class="rounded-t-lg"
                src="https://wallpapercave.com/wp/wp2616571.jpg"
                alt=""
              />
            </a>
            <div class="p-5">
              <a href="diabetes">
                <h5 class="mb-2 text-2xl font-bold text-center tracking-tight text-gray-900 dark:text-white">
                  Diabetes Disease Prediction
                </h5>
              </a>
              <p className="mb-3 font-normal text-gray-700 dark:text-gray-400 text-center">
                Diabetes is a metabolic disorder characterized by high blood
                sugar levels, either due to insufficient insulin production or
                ineffective use of insulin by the body. Risk factors include
                obesity, unhealthy diet, physical inactivity, and genetic
                predisposition.
              </p>
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
        </div>
      </div>
    </div>
  );
}

export default HomePage;
