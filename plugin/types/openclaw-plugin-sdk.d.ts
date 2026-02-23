/**
 * Type shim for `openclaw/plugin-sdk`.
 *
 * The runtime module is provided by the OpenClaw host process and is never
 * installed as an npm dependency. This ambient declaration lets TypeScript
 * tooling (tsc, language-server) resolve the import without a published
 * package, while keeping the surface minimal and forward-compatible.
 *
 * If OpenClaw ships a proper `@openclaw/plugin-sdk` package in the future,
 * remove this file and add the real dependency to package.json.
 */
declare module 'openclaw/plugin-sdk' {
  /**
   * The API surface handed to a plugin's `register()` function by the
   * OpenClaw host.  Only the members actually used by this plugin are
   * declared here; the host may provide additional members at runtime.
   */
  export interface OpenClawPluginApi {
    /**
     * Arbitrary plugin configuration object supplied via the host config
     * (e.g. `pluginConfig` in `openclaw.plugin.json`).  Cast to the
     * expected shape inside the plugin.
     */
    readonly pluginConfig?: unknown;

    /**
     * Register a lifecycle-hook handler.
     *
     * @param event  - Hook name (e.g. `'before_prompt_build'`).
     * @param handler - Async function that receives the event payload and
     *                  a mutable context object, and returns an optional
     *                  partial result merged by the host.
     */
    on(
      event: string,
      handler: (event: unknown, ctx: Record<string, unknown>) => unknown,
    ): void;
  }
}
