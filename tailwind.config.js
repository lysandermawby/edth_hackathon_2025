/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f4ff',
          100: '#e0eaff',
          500: '#667eea',
          600: '#5a6fd8',
          700: '#4c5cc2',
          800: '#3f4aad',
          900: '#3239a3',
        },
        secondary: {
          500: '#764ba2',
          600: '#6a4296',
          700: '#5e3989',
        }
      },
      animation: {
        'spin-slow': 'spin 2s linear infinite',
      }
    },
  },
  plugins: [],
}