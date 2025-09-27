/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Cyberpunk color palette - pure neon on black
        cyber: {
          black: "#000000",
          dark: "#0a0a0a", 
          bg: "#111111",
          surface: "#1a1a1a",
          border: "#333333",
          text: "#ffffff",
          muted: "#999999",
          disabled: "#555555",
        },
        neon: {
          cyan: "#00ffff",
          pink: "#ff0080", 
          green: "#00ff41",
          yellow: "#ffff00",
          purple: "#8000ff",
          blue: "#0080ff",
          red: "#ff4040",
          orange: "#ff8000",
        },
        // Legacy colors mapped to neon equivalents for compatibility
        primary: {
          50: "#e6ffff",
          100: "#ccffff",
          200: "#99ffff",
          300: "#66ffff", 
          400: "#33ffff",
          500: "#00ffff", // Neon cyan
          600: "#00cccc", 
          700: "#009999",
          800: "#006666",
          900: "#003333",
        },
        secondary: {
          50: "#ffe6f5",
          100: "#ffcceb",
          200: "#ff99d6",
          300: "#ff66c2",
          400: "#ff33ad",
          500: "#ff0080", // Neon pink
          600: "#cc0066",
          700: "#99004d",
          800: "#660033",
          900: "#33001a",
        },
        accent: {
          50: "#e6fff2",
          100: "#ccffe6",
          200: "#99ffcc",
          300: "#66ffb3",
          400: "#33ff99",
          500: "#00ff41", // Neon green
          600: "#00cc34",
          700: "#009928",
          800: "#00661b",
          900: "#00330d",
        },
        success: {
          400: "#00ff41",
          500: "#00cc34",
          600: "#009928",
        },
        warning: {
          400: "#ffff00",
          500: "#cccc00",
          600: "#999900",
        },
        // Tactical colors mapped to cyberpunk
        tactical: {
          bg: "#000000",
          surface: "#111111",
          border: "#333333",
          text: "#ffffff",
          muted: "#999999",
          glow: "#00ffff",
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Courier New', 'monospace'],
        sans: ['JetBrains Mono', 'Courier New', 'monospace'], // Everything monospace for cyberpunk feel
      },
      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1.2rem', letterSpacing: '0.05em' }],
        'sm': ['0.875rem', { lineHeight: '1.4rem', letterSpacing: '0.05em' }],
        'base': ['1rem', { lineHeight: '1.6rem', letterSpacing: '0.025em' }],
        'lg': ['1.125rem', { lineHeight: '1.8rem', letterSpacing: '0.025em' }],
        'xl': ['1.25rem', { lineHeight: '2rem', letterSpacing: '0.025em' }],
        '2xl': ['1.5rem', { lineHeight: '2.4rem', letterSpacing: '0.025em' }],
        '3xl': ['1.875rem', { lineHeight: '2.8rem', letterSpacing: '0.025em' }],
        '4xl': ['2.25rem', { lineHeight: '3.2rem', letterSpacing: '0.025em' }],
      },
      borderRadius: {
        'none': '0px', // No rounded corners - everything square/rectangular
        'xs': '0px',
        'sm': '0px', 
        'DEFAULT': '0px',
        'md': '0px',
        'lg': '0px',
        'xl': '0px',
        '2xl': '0px',
        '3xl': '0px',
        'full': '0px',
      },
      boxShadow: {
        // Neon glow effects
        neon: "0 0 5px currentColor, 0 0 10px currentColor",
        "neon-sm": "0 0 2px currentColor, 0 0 4px currentColor",
        "neon-lg": "0 0 10px currentColor, 0 0 20px currentColor, 0 0 30px currentColor",
        "neon-xl": "0 0 15px currentColor, 0 0 30px currentColor, 0 0 45px currentColor",
        // Cyberpunk specific
        cyber: "0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff",
        "cyber-pink": "0 0 10px #ff0080, 0 0 20px #ff0080, 0 0 30px #ff0080",
        "cyber-green": "0 0 10px #00ff41, 0 0 20px #00ff41, 0 0 30px #00ff41",
        // Subtle inner shadows for depth
        inner: "inset 0 2px 4px rgba(0, 0, 0, 0.9)",
        "inner-glow": "inset 0 0 10px rgba(0, 255, 255, 0.1)",
      },
      animation: {
        'pulse-neon': 'pulse-neon 2s ease-in-out infinite',
        'flicker': 'flicker 3s linear infinite',
        'scan': 'scan 2s linear infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        'pulse-neon': {
          '0%, 100%': { 
            textShadow: '0 0 5px currentColor, 0 0 10px currentColor, 0 0 15px currentColor',
            filter: 'brightness(1)'
          },
          '50%': { 
            textShadow: '0 0 2px currentColor, 0 0 5px currentColor, 0 0 8px currentColor',
            filter: 'brightness(0.8)'
          },
        },
        flicker: {
          '0%, 19%, 21%, 23%, 25%, 54%, 56%, 100%': { 
            opacity: '1',
            filter: 'brightness(1)',
          },
          '20%, 24%, 55%': { 
            opacity: '0.7',
            filter: 'brightness(0.7)',
          },
        },
        scan: {
          '0%': { transform: 'translateX(-100%)', opacity: '0' },
          '50%': { opacity: '1' },
          '100%': { transform: 'translateX(100%)', opacity: '0' },
        },
        glow: {
          '0%': { 
            textShadow: '0 0 5px currentColor, 0 0 10px currentColor, 0 0 15px currentColor',
          },
          '100%': { 
            textShadow: '0 0 10px currentColor, 0 0 20px currentColor, 0 0 30px currentColor',
          },
        },
      },
      backgroundImage: {
        'scan-lines': 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0, 255, 255, 0.03) 2px, rgba(0, 255, 255, 0.03) 4px)',
        'grid-cyber': 'linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px)',
      },
      backgroundSize: {
        'grid': '20px 20px',
      },
    },
  },
  plugins: [],
};