/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink:    { DEFAULT: '#12151A', soft: '#3A424E', mute: '#6B7480', faint: '#9AA3AF' },
        paper:  { DEFAULT: '#FFFFFF', warm: '#FAFAF8', sunk: '#F4F4F1' },
        rule:   { DEFAULT: '#E6E7E3', strong: '#D2D4CE' },
        signal: { DEFAULT: '#C8322B', soft: '#F3DEDC', deep: '#8E211C' },
        calm:   { DEFAULT: '#3E7CA6', soft: '#DDE8F0' },
        gold:   { DEFAULT: '#B07D2B', soft: '#F2E7D2' },
        moss:   { DEFAULT: '#4A7C59', soft: '#DFE9E1' },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Helvetica Neue', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'SF Mono', 'Menlo', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      maxWidth: { content: '68rem', prose: '38rem' },
      transitionTimingFunction: { smooth: 'cubic-bezier(0.22, 0.61, 0.36, 1)' },
    },
  },
  plugins: [],
}
