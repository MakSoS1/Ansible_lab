/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        dance: {
          bg: '#0a0a1a',
          card: '#1a1a2e',
          accent: '#6200ea',
          perfect: '#ffd700',
          super: '#ff6f00',
          good: '#4caf50',
          ok: '#2196f3',
          x: '#f44336',
          yeah: '#ffd700',
        },
      },
      animation: {
        'pop-in': 'popIn 0.3s ease-out',
        'pulse-glow': 'pulseGlow 0.5s ease-in-out infinite alternate',
      },
      keyframes: {
        popIn: {
          '0%': { transform: 'scale(0.5)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        pulseGlow: {
          '0%': { opacity: '0.7', textShadow: '0 0 10px currentColor' },
          '100%': { opacity: '1', textShadow: '0 0 30px currentColor' },
        },
      },
    },
  },
  plugins: [],
}
