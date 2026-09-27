/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink:    { DEFAULT: '#F1F0EB', soft: '#C4C7CA', mute: '#A5AAB0', faint: '#9299A0' },
        paper:  { DEFAULT: '#101113', warm: '#191B1E', sunk: '#22262B' },
        rule:   { DEFAULT: '#30363D', strong: '#4D555F' },
        signal: { DEFAULT: '#B2A29D', soft: '#2C2524', deep: '#C8B4AD' },
        calm:   { DEFAULT: '#ACB7C1', soft: '#252A30' },
        gold:   { DEFAULT: '#B6BEC8', soft: '#282D34' },
        moss:   { DEFAULT: '#9AAFA5', soft: '#242E29' },
      },
      fontFamily: {
        sans: ['ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Helvetica Neue', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'SF Mono', 'Menlo', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      maxWidth: { content: '80rem', prose: '38rem' },
      transitionTimingFunction: { smooth: 'cubic-bezier(0.22, 0.61, 0.36, 1)' },
    },
  },
  plugins: [],
}
