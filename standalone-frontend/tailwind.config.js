/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          500: "#3B82F6",
          600: "#2563EB",
          700: "#1D4ED8",
        },
        secondary: {
          500: "#8B5CF6",
          600: "#7C3AED",
          700: "#6D28D9",
        },
      },
    },
  },
  plugins: [],
};
