export interface GraphitiCliStatus {
  plugin: string;
  healthy: boolean;
  detail: string;
}

export const graphitiCliStatus = (): GraphitiCliStatus => ({
  plugin: 'bicameral',
  healthy: true,
  detail: 'Plugin scaffold is installed. Use /bicameral status in runtime for live checks.',
});
