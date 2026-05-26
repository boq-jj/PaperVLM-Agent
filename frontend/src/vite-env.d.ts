/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_PAPERVLM_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
