import { defineConfig } from 'slidev/config'
import container from 'markdown-it-container'

const CALLOUTS = [
  { type: 'tip', label: 'Tip' },
  { type: 'info', label: 'Info' },
  { type: 'warning', label: 'Warning' },
]

export default defineConfig({
  markdown: {
    config(md) {
      CALLOUTS.forEach(({ type, label }) => {
        md.use(container, type, {
          render(tokens, idx) {
            const token = tokens[idx]
            if (token.nesting === 1) {
              const rawInfo = token.info.trim()
              let params = rawInfo.slice(type.length).trim()
              let title = label

              if (params.startsWith('{') && params.endsWith('}')) {
                const attrs = params.slice(1, -1)
                const match =
                  attrs.match(/title\s*=\s*"([^"]*)"/) || attrs.match(/title\s*=\s*'([^']*)'/)
                if (match && match[1]) {
                  title = match[1]
                }
              } else if (params.length > 0) {
                title = params
              }

              return `<div class="callout callout-${type}"><div class="callout-title">${md.utils.escapeHtml(title)}</div><div class="callout-body">\n`
            }
            return '</div></div>\n'
          },
        })
      })
    },
  },
  vite: {
    base: process.env.BASE_PATH ?? '/',
  },
})
