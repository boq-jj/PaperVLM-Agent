import React from "react";
import ReactDOM from "react-dom/client";
import {
  FileText,
  Image,
  Loader2,
  Plus,
  Send,
  Settings2,
  X,
} from "lucide-react";
import "./styles.css";

type Mode = "text" | "visual";
type BackendStatus = "idle" | "processing" | "ready" | "error";
type AnswerLanguage = "中文" | "English";
type Role = "user" | "assistant";

type RetrievedChunk = {
  rank?: number;
  score?: number;
  chunk_id?: string;
  page_id?: number;
  preview?: string;
};

type ChatMessage = {
  id: string;
  role: Role;
  mode?: Mode;
  content: string;
  evidence?: RetrievedChunk[];
};

type AskResponse = {
  answer?: string;
  retrieved_chunks?: RetrievedChunk[];
};

const DEFAULT_API_BASE_URL = import.meta.env.VITE_PAPERVLM_API_BASE_URL?.replace(/\/$/, "") ?? "";

class UserFacingError extends Error {
  constructor(
    message: string,
    readonly detail = "",
  ) {
    super(message);
    this.name = "UserFacingError";
  }
}

const defaultTextSettings = {
  llmBackend: "qwen-vl",
  llmModelName: "qwen3-vl-flash",
  answerLanguage: "中文" as AnswerLanguage,
  topK: 5,
  maxContextChars: 4000,
  maxNewTokens: 512,
  temperature: 0.2,
};

const defaultVisualSettings = {
  answerLanguage: "中文" as AnswerLanguage,
  topK: 5,
  maxContextChars: 3000,
  maxNewTokens: 768,
  temperature: 0.2,
};

function App() {
  const [paperId, setPaperId] = React.useState("");
  const [paperName, setPaperName] = React.useState("");
  const [status, setStatus] = React.useState<BackendStatus>("idle");
  const [statusText, setStatusText] = React.useState("");
  const [question, setQuestion] = React.useState("");
  const [mode, setMode] = React.useState<Mode>("text");
  const [apiBaseUrl, setApiBaseUrl] = React.useState(() => {
    return localStorage.getItem("papervlm_api_base_url") || DEFAULT_API_BASE_URL;
  });
  const [textSettings, setTextSettings] = React.useState(defaultTextSettings);
  const [visualSettings, setVisualSettings] = React.useState(defaultVisualSettings);
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [visualImage, setVisualImage] = React.useState<File | null>(null);
  const [isAsking, setIsAsking] = React.useState(false);
  const [showSettings, setShowSettings] = React.useState(false);
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);
  const imageInputRef = React.useRef<HTMLInputElement | null>(null);
  const chatEndRef = React.useRef<HTMLDivElement | null>(null);

  const hasPaper = Boolean(paperId);
  const normalizedApiBaseUrl = apiBaseUrl.trim().replace(/\/$/, "");
  const backendConnected = Boolean(normalizedApiBaseUrl);

  React.useEffect(() => {
    const value = apiBaseUrl.trim();
    if (value) {
      localStorage.setItem("papervlm_api_base_url", value);
    } else {
      localStorage.removeItem("papervlm_api_base_url");
    }
  }, [apiBaseUrl]);

  React.useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isAsking]);

  async function processPdf(file: File | null) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setStatus("error");
      setStatusText("请上传 PDF 文件。");
      return;
    }

    setStatus("processing");
    setPaperName(file.name);
    setMessages([]);
    setStatusText("正在处理 PDF，生成文本块和检索索引...");

    if (!backendConnected) {
      const demoPaperId = file.name.replace(/\.pdf$/i, "") || "uploaded_paper";
      setPaperId(demoPaperId);
      setStatus("ready");
      setStatusText(
        [
          "已选择 PDF，但当前只部署了 Netlify 前端。",
          `paper_id: ${demoPaperId}`,
          "未配置后端 API 地址，因此无法解析 PDF、构建索引或生成真实回答。",
          "请点击齿轮设置，填写 Python 后端 API 地址。",
        ].join("\n"),
      );
      setMessages([
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "已收到 PDF 文件，但当前网页没有连接后端，所以不能执行论文解析和真实问答。请点击齿轮设置，填写 Python 后端 API 地址后再提问。",
        },
      ]);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("pdf", file);
      formData.append("chunk_size", "800");
      formData.append("chunk_overlap", "150");

      const response = await fetch(`${normalizedApiBaseUrl}/api/process-pdf`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw await buildApiError(response, "/api/process-pdf");
      }
      ensureJsonResponse(response, "/api/process-pdf");

      const data = await response.json();
      setPaperId(data.paper_id ?? "");
      setStatus("ready");
      setStatusText(data.status ?? "PDF 处理完成。");
      setMessages([
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "论文已处理完成。现在可以开始问答。",
        },
      ]);
    } catch (error) {
      setStatus("error");
      setStatusText(formatErrorForStatus(error));
    }
  }

  async function askQuestion() {
    const trimmedQuestion = question.trim();
    if (!hasPaper) {
      setMessages([
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "请先上传并处理 PDF。",
        },
      ]);
      return;
    }
    if (!trimmedQuestion) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      mode,
      content: trimmedQuestion,
    };

    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setIsAsking(true);

    if (!backendConnected) {
      window.setTimeout(() => {
        setMessages((current) => [
          ...current,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            mode,
            content: [
              "当前无法生成真实回答。",
              "",
              "原因：Netlify 现在只部署了前端页面，尚未连接 PaperVLM-Agent 的 Python 后端。",
              "",
              "请点击齿轮设置，填写后端 API 地址，例如：",
              "https://你的后端地址",
            ].join("\n"),
          },
        ]);
        setIsAsking(false);
      }, 200);
      return;
    }

    try {
      let response: Response;
      if (mode === "text") {
        response = await fetch(`${normalizedApiBaseUrl}/api/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            paper_id: paperId,
            query: trimmedQuestion,
            llm_backend: textSettings.llmBackend,
            llm_model_name: textSettings.llmModelName,
            answer_language: textSettings.answerLanguage,
            top_k: textSettings.topK,
            max_context_chars: textSettings.maxContextChars,
            max_new_tokens: textSettings.maxNewTokens,
            temperature: textSettings.temperature,
          }),
        });
      } else {
        const formData = new FormData();
        formData.append("paper_id", paperId);
        formData.append("query", trimmedQuestion);
        formData.append("answer_language", visualSettings.answerLanguage);
        formData.append("top_k", String(visualSettings.topK));
        formData.append("max_context_chars", String(visualSettings.maxContextChars));
        formData.append("max_new_tokens", String(visualSettings.maxNewTokens));
        formData.append("temperature", String(visualSettings.temperature));
        if (visualImage) formData.append("image", visualImage);

        response = await fetch(`${normalizedApiBaseUrl}/api/ask-visual`, {
          method: "POST",
          body: formData,
        });
      }

      const data = await readAskResponse(response);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          mode,
          content: data.answer ?? "后端没有返回回答。",
          evidence: data.retrieved_chunks ?? [],
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          mode,
          content: formatErrorForChat(error),
        },
      ]);
    } finally {
      setIsAsking(false);
    }
  }

  async function readAskResponse(response: Response): Promise<AskResponse> {
    if (!response.ok) {
      throw await buildApiError(response, mode === "text" ? "/api/ask" : "/api/ask-visual");
    }
    ensureJsonResponse(response, mode === "text" ? "/api/ask" : "/api/ask-visual");
    return (await response.json()) as AskResponse;
  }

  async function buildApiError(response: Response, endpoint: string): Promise<UserFacingError> {
    const text = await response.text();
    if (looksLikeHtml(text)) {
      return new UserFacingError(
        "后端地址不可用。请在齿轮设置中填写真正的 Python 后端地址，不要填写 Netlify 前端网址。",
        `${endpoint} returned HTML, HTTP ${response.status}`,
      );
    }
    return new UserFacingError(
      "后端请求失败。请检查后端服务是否正在运行，以及接口是否允许跨域访问。",
      text || `${endpoint} failed, HTTP ${response.status}`,
    );
  }

  function ensureJsonResponse(response: Response, endpoint: string) {
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      throw new UserFacingError(
        "后端返回格式不正确。请确认该地址是 PaperVLM-Agent Python 后端。",
        `${endpoint} content-type=${contentType || "<empty>"}`,
      );
    }
  }

  function formatErrorForStatus(error: unknown) {
    if (error instanceof UserFacingError) {
      console.warn(error.detail);
      return `PDF 处理失败。\n${error.message}`;
    }
    return `PDF 处理失败。\n${String(error)}`;
  }

  function formatErrorForChat(error: unknown) {
    if (error instanceof UserFacingError) {
      console.warn(error.detail);
      return ["回答失败。", "", error.message].join("\n");
    }
    return `回答失败。\n${String(error)}`;
  }

  function looksLikeHtml(text: string) {
    const trimmed = text.trim().toLowerCase();
    return trimmed.startsWith("<!doctype html") || trimmed.startsWith("<html");
  }

  function onDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    processPdf(event.dataTransfer.files.item(0));
  }

  return (
    <main
      className={hasPaper ? "app-shell ready" : "app-shell"}
      onDragOver={(event) => event.preventDefault()}
      onDrop={onDrop}
    >
      {!hasPaper && (
        <section className="hero">
          <h1>PaperVLM-Agent</h1>
          <p>科研论文中文问答助手</p>
          <Composer
            question={question}
            setQuestion={setQuestion}
            mode={mode}
            onAsk={askQuestion}
            onUpload={() => fileInputRef.current?.click()}
            onSettings={() => setShowSettings(true)}
            isAsking={isAsking}
            disabled={!hasPaper}
          />
          {statusText && <pre className={`status ${status}`}>{statusText}</pre>}
        </section>
      )}

      {hasPaper && (
        <>
          <section className="workspace chat-layout">
            <header className="workspace-header">
              <div>
                <span className="eyebrow">当前论文</span>
                <h2>{paperName || paperId}</h2>
              </div>
              <div className="workspace-actions">
                <button className="small-action" type="button" onClick={() => fileInputRef.current?.click()}>
                  <Plus size={18} />
                  更换 PDF
                </button>
              </div>
            </header>

            <section className="chat-window">
              <div className="chat-meta">
              <div>
                <strong>{mode === "text" ? "文本问答" : "视觉问答"}</strong>
                <span>
                  {backendConnected
                    ? mode === "text"
                      ? `回答语言：${textSettings.answerLanguage}`
                      : `回答语言：${visualSettings.answerLanguage}${visualImage ? ` · 图像：${visualImage.name}` : ""}`
                    : "后端未连接，无法生成真实回答"}
                </span>
              </div>
                <FileText size={20} />
              </div>

              <div className="chat-messages">
                {messages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))}
                {isAsking && (
                  <div className="message-row assistant">
                    <div className="message-bubble">
                      <Loader2 className="spin" size={18} />
                      正在生成回答...
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
            </section>
          </section>

          <footer className="bottom-composer">
            <Composer
              question={question}
              setQuestion={setQuestion}
              mode={mode}
              onAsk={askQuestion}
              onUpload={() => fileInputRef.current?.click()}
              onSettings={() => setShowSettings(true)}
              isAsking={isAsking}
              disabled={false}
            />
          </footer>
        </>
      )}

      {showSettings && (
        <SettingsDrawer
          mode={mode}
          setMode={setMode}
          apiBaseUrl={apiBaseUrl}
          setApiBaseUrl={setApiBaseUrl}
          textSettings={textSettings}
          setTextSettings={setTextSettings}
          visualSettings={visualSettings}
          setVisualSettings={setVisualSettings}
          visualImage={visualImage}
          onPickImage={() => imageInputRef.current?.click()}
          onClose={() => setShowSettings(false)}
        />
      )}

      <input
        ref={fileInputRef}
        className="hidden-input"
        type="file"
        accept="application/pdf,.pdf"
        onChange={(event) => processPdf(event.currentTarget.files?.item(0) ?? null)}
      />
      <input
        ref={imageInputRef}
        className="hidden-input"
        type="file"
        accept="image/png,image/jpeg,image/webp"
        onChange={(event) => setVisualImage(event.currentTarget.files?.item(0) ?? null)}
      />
    </main>
  );
}

function Composer(props: {
  question: string;
  setQuestion: (value: string) => void;
  mode: Mode;
  onAsk: () => void;
  onUpload: () => void;
  onSettings: () => void;
  isAsking: boolean;
  disabled: boolean;
}) {
  return (
    <div className="composer">
      <button className="icon-button" type="button" aria-label="上传 PDF" onClick={props.onUpload}>
        <Plus size={24} />
      </button>
      <textarea
        value={props.question}
        placeholder={props.disabled ? "上传 PDF 后开始提问" : "请输入关于这篇论文的问题..."}
        onChange={(event) => props.setQuestion(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            props.onAsk();
          }
        }}
      />
      <button
        className="settings-button"
        type="button"
        aria-label="打开问答设置"
        onClick={props.onSettings}
      >
        <Settings2 size={21} />
      </button>
      <button className="send-button" type="button" onClick={props.onAsk} disabled={props.isAsking}>
        {props.isAsking ? <Loader2 className="spin" size={20} /> : <Send size={20} />}
      </button>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  return (
    <div className={`message-row ${message.role}`}>
      <div className="message-bubble">
        {message.mode && <span className="message-mode">{message.mode === "text" ? "文本" : "图像"}</span>}
        <pre>{message.content}</pre>
        {message.evidence && message.evidence.length > 0 && <EvidenceList chunks={message.evidence} />}
      </div>
    </div>
  );
}

function SettingsDrawer(props: {
  mode: Mode;
  setMode: (value: Mode) => void;
  apiBaseUrl: string;
  setApiBaseUrl: (value: string) => void;
  textSettings: typeof defaultTextSettings;
  setTextSettings: (value: typeof defaultTextSettings) => void;
  visualSettings: typeof defaultVisualSettings;
  setVisualSettings: (value: typeof defaultVisualSettings) => void;
  visualImage: File | null;
  onPickImage: () => void;
  onClose: () => void;
}) {
  return (
    <div className="settings-overlay" onClick={props.onClose}>
      <aside className="settings-drawer" onClick={(event) => event.stopPropagation()}>
        <header>
          <div>
            <span className="eyebrow">问答设置</span>
            <h3>配置当前对话</h3>
          </div>
          <button className="close-button" type="button" onClick={props.onClose} aria-label="关闭设置">
            <X size={20} />
          </button>
        </header>

        <div className="mode-card">
          <button className={props.mode === "text" ? "active" : ""} type="button" onClick={() => props.setMode("text")}>
            文本问答
          </button>
          <button className={props.mode === "visual" ? "active" : ""} type="button" onClick={() => props.setMode("visual")}>
            视觉问答
          </button>
        </div>

        <div className="settings-grid api-settings">
          <InputRow
            label="后端 API 地址"
            value={props.apiBaseUrl}
            onChange={props.setApiBaseUrl}
          />
          <p className="settings-hint">
            例如：https://your-backend.example.com。这里必须填写 Python 后端地址，不要填写 Netlify 前端网址。
            前端会自动调用 /api/process-pdf、/api/ask 和 /api/ask-visual。
          </p>
        </div>

        {props.mode === "text" ? (
          <div className="settings-grid">
            <SelectRow
              label="回答语言"
              value={props.textSettings.answerLanguage}
              options={["中文", "English"]}
              onChange={(value) =>
                props.setTextSettings({ ...props.textSettings, answerLanguage: value as AnswerLanguage })
              }
            />
            <SelectRow
              label="LLM 后端"
              value={props.textSettings.llmBackend}
              options={["qwen-vl", "mock"]}
              onChange={(value) => props.setTextSettings({ ...props.textSettings, llmBackend: value })}
            />
            <InputRow
              label="模型"
              value={props.textSettings.llmModelName}
              onChange={(value) => props.setTextSettings({ ...props.textSettings, llmModelName: value })}
            />
            <NumberRow
              label="top_k"
              value={props.textSettings.topK}
              onChange={(value) => props.setTextSettings({ ...props.textSettings, topK: value })}
            />
            <NumberRow
              label="上下文长度"
              value={props.textSettings.maxContextChars}
              onChange={(value) => props.setTextSettings({ ...props.textSettings, maxContextChars: value })}
            />
            <NumberRow
              label="temperature"
              value={props.textSettings.temperature}
              step={0.1}
              onChange={(value) => props.setTextSettings({ ...props.textSettings, temperature: value })}
            />
          </div>
        ) : (
          <div className="settings-grid">
            <SelectRow
              label="回答语言"
              value={props.visualSettings.answerLanguage}
              options={["中文", "English"]}
              onChange={(value) =>
                props.setVisualSettings({ ...props.visualSettings, answerLanguage: value as AnswerLanguage })
              }
            />
            <button className="image-picker" type="button" onClick={props.onPickImage}>
              <Image size={18} />
              {props.visualImage ? props.visualImage.name : "可选：上传页面或图表图片"}
            </button>
            <NumberRow
              label="top_k"
              value={props.visualSettings.topK}
              onChange={(value) => props.setVisualSettings({ ...props.visualSettings, topK: value })}
            />
            <NumberRow
              label="上下文长度"
              value={props.visualSettings.maxContextChars}
              onChange={(value) => props.setVisualSettings({ ...props.visualSettings, maxContextChars: value })}
            />
            <NumberRow
              label="temperature"
              value={props.visualSettings.temperature}
              step={0.1}
              onChange={(value) => props.setVisualSettings({ ...props.visualSettings, temperature: value })}
            />
          </div>
        )}
      </aside>
    </div>
  );
}

function SelectRow(props: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <select value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        {props.options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function InputRow(props: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <input value={props.value} onChange={(event) => props.onChange(event.target.value)} />
    </label>
  );
}

function NumberRow(props: {
  label: string;
  value: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <input
        type="number"
        step={props.step ?? 1}
        value={props.value}
        onChange={(event) => props.onChange(Number(event.target.value))}
      />
    </label>
  );
}

function EvidenceList({ chunks }: { chunks: RetrievedChunk[] }) {
  return (
    <div className="evidence-list">
      {chunks.map((chunk, index) => (
        <article key={`${chunk.chunk_id ?? "chunk"}-${index}`} className="evidence-card">
          <div>
            <strong>第 {chunk.rank ?? index + 1} 条证据</strong>
            <span>page {chunk.page_id ?? "-"}</span>
          </div>
          <p>{chunk.preview}</p>
          <small>
            {chunk.chunk_id ?? "unknown"} · score {chunk.score?.toFixed(4) ?? "-"}
          </small>
        </article>
      ))}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
