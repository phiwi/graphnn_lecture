declare module 'slidev/config' {
  // Minimal fallback typing for Slidev config helper.
  export function defineConfig<T>(config: T): T
}

declare module 'markdown-it-container' {
  type Plugin = (md: unknown, ...params: unknown[]) => void
  const plugin: Plugin
  export default plugin
}
