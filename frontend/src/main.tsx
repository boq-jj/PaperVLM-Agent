import React from "react";
import ReactDOM from "react-dom/client";
import {
  FileText,
  Image,
  Loader2,
  MessageSquareText,
  Plus,
  Send,
  Settings2,
} from "lucide-react";
import "./styles.css";

type Mode = "text" | "visual";
type BackendStatus = "idle" | "processing" | "ready" | "error";
type AnswerLanguage = "中文" | "English";

type RetrievedChunk = {
  rank?: number;
  score?: number;
  chunk_id?: string;
  page_id?: number;
  preview?: string;
};

type AskResponse = {
  answer?: string;
  retrieved_chunks?: RetrievedChunk[];
};

const API_BASE_URL = import.meta.env.VITE_PAPERVLM_API_BASE_URL?.replace(/\/$/, "") ?? "";

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
  const [textSettings, setTextSettings] = React.useState(defaultTextSettings);
  const [visualSettings, setVisualSettings] = React.useState(defaultVisualSettings);
  const [answer, setAnswer] = React.useState("");
  const [evidence, setEvidence] = React.useState<RetrievedChunk[]>([]);
  const [visualImage, setVisualImage] = React.useState<File | null>(null);
  const [isAsking, setIsAsking] = React.useState(false);
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);
  const imageInputRef = React.useRef<HTMLInputElement | null>(null);

  const hasPaper = Boolean(paperId);

  async function processPdf(file: File | null) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setStatus("error");
      setStatusText("请上传 PDF 文件。");
      return;
    }

    setStatus("processing");
    setPaperName(file.name);
    setStatusText("正在处理 PDF，生成文本块和检索索引...");

    if (!API_BASE_URL) {
      const demoPaperId = file.name.replace(/\.pdf$/i, "") || "uploaded_paper";
      setPaperId(demoPaperId);
      setStatus("ready");
      setStatusText(
        [
          "前端演示模式已就绪。",
          `paper_id: ${demoPaperId}`,
          "当前未配置 VITE_PAPERVLM_API_BASE_URL，因此不会调用 Python 后端。",
          "部署后可把该环境变量设置为后端 API 地址。",
        ].join("\n"),
      );
      return;
    }

    try {
      const formData = new FormData();
      formData.append("pdf", file);
      formData.append("chunk_size", "800");
      formData.append("chunk_overlap", "150");

      const response = await fetch(`${API_BASE_URL}/api/process-pdf`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = await response.json();
      setPaperId(data.paper_id ?? "");
      setStatus("ready");
      setStatusText(data.status ?? "PDF 处理完成。");
    } catch (error) {
      setStatus("error");
      setStatusText(`PDF 处理失败：${String(error)}`);
    }
  }

  async function askQuestion() {
    const trimmedQuestion = question.trim();
    if (!hasPaper) {
      setAnswer("请先上传并处理 PDF。");
      return;
    }
    if (!trimmedQuestion) {
      setAnswer("请输入问题。");
      return;
    }

    setIsAsking(true);
    setAnswer("正在生成回答...");
    setEvidence([]);

    if (!API_BASE_URL) {
      setAnswer(
        mode === "text"
          ? `回答：这是前端演示回答。你问的是：“${trimmedQuestion}”。\n\n支持页码：请连接 Python 后端后查看真实检索页码。\n\n推理：Netlify 前端已完成，真实 PDF 解析、FAISS 检索和 Qwen 调用需要后端 API。`
          : `回答：这是前端演示的视觉问答结果。\n\n支持页码：请连接 Python 后端后查看。\n\n视觉证据：${visualImage ? visualImage.name : "未上传图片，将由后端自动选择检索页截图。"}\n\n推理：图像理解需要后端调用 qwen3-vl-flash。`,
      );
      setEvidence([
        {
          rank: 1,
          score: 0.88,
          chunk_id: "frontend_demo_chunk",
          page_id: 1,
          preview: "这里会显示 Python 后端返回的检索证据预览。",
        },
      ]);
      setIsAsking(false);
      return;
    }

    try {
      if (mode === "text") {
        const response = await fetch(`${API_BASE_URL}/api/ask`, {
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
        await handleAskResponse(response);
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

        const response = await fetch(`${API_BASE_URL}/api/ask-visual`, {
          method: "POST",
          body: formData,
        });
        await handleAskResponse(response);
      }
    } catch (error) {
      setAnswer(`回答失败：${String(error)}`);
    } finally {
      setIsAsking(false);
    }
  }

  async function handleAskResponse(response: Response) {
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const data = (await response.json()) as AskResponse;
    setAnswer(data.answer ?? "后端没有返回回答。");
    setEvidence(data.retrieved_chunks ?? []);
  }

  function onDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    processPdf(event.dataTransfer.files.item(0));
  }

  return (
    <main className={hasPaper ? "app-shell ready" : "app-shell"} onDragOver={(event) => event.preventDefault()} onDrop={onDrop}>
      {!hasPaper && (
        <section className="hero">
          <h1>PaperVLM-Agent</h1>
          <p>科研论文中文问答助手</p>
          <Composer
            question={question}
            setQuestion={setQuestion}
            mode={mode}
            setMode={setMode}
            onAsk={askQuestion}
            onUpload={() => fileInputRef.current?.click()}
            isAsking={isAsking}
            disabled={!hasPaper}
          />
          {statusText && <pre className={`status ${status}`}>{statusText}</pre>}
        </section>
      )}

      {hasPaper && (
        <>
          <section className="workspace">
            <header className="workspace-header">
              <div>
                <span className="eyebrow">当前论文</span>
                <h2>{paperName || paperId}</h2>
              </div>
              <pre className={`status compact ${status}`}>{statusText}</pre>
            </header>

            <div className="qa-grid">
              <Panel
                active={mode === "text"}
                icon={<MessageSquareText size={20} />}
                title="文本问答"
                onClick={() => setMode("text")}
              >
                <SettingsBlock title="文本问答设置">
                  <SelectRow
                    label="回答语言"
                    value={textSettings.answerLanguage}
                    options={["中文", "English"]}
                    onChange={(value) =>
                      setTextSettings({ ...textSettings, answerLanguage: value as AnswerLanguage })
                    }
                  />
                  <SelectRow
                    label="LLM 后端"
                    value={textSettings.llmBackend}
                    options={["qwen-vl", "mock"]}
                    onChange={(value) => setTextSettings({ ...textSettings, llmBackend: value })}
                  />
                  <InputRow
                    label="模型"
                    value={textSettings.llmModelName}
                    onChange={(value) => setTextSettings({ ...textSettings, llmModelName: value })}
                  />
                  <NumberRow
                    label="top_k"
                    value={textSettings.topK}
                    onChange={(value) => setTextSettings({ ...textSettings, topK: value })}
                  />
                  <NumberRow
                    label="上下文长度"
                    value={textSettings.maxContextChars}
                    onChange={(value) => setTextSettings({ ...textSettings, maxContextChars: value })}
                  />
                </SettingsBlock>
              </Panel>

              <Panel active={mode === "visual"} icon={<Image size={20} />} title="视觉问答" onClick={() => setMode("visual")}>
                <SettingsBlock title="视觉问答设置">
                  <SelectRow
                    label="回答语言"
                    value={visualSettings.answerLanguage}
                    options={["中文", "English"]}
                    onChange={(value) =>
                      setVisualSettings({ ...visualSettings, answerLanguage: value as AnswerLanguage })
                    }
                  />
                  <button className="image-picker" type="button" onClick={() => imageInputRef.current?.click()}>
                    <Image size={18} />
                    {visualImage ? visualImage.name : "可选：上传页面或图表图片"}
                  </button>
                  <NumberRow
                    label="top_k"
                    value={visualSettings.topK}
                    onChange={(value) => setVisualSettings({ ...visualSettings, topK: value })}
                  />
                  <NumberRow
                    label="上下文长度"
                    value={visualSettings.maxContextChars}
                    onChange={(value) => setVisualSettings({ ...visualSettings, maxContextChars: value })}
                  />
                </SettingsBlock>
              </Panel>
            </div>

            <section className="answer-board">
              <div>
                <h3>回答</h3>
                <pre>{answer || "上传论文并输入问题后，这里会显示回答。"}</pre>
              </div>
              <div>
                <h3>检索证据</h3>
                <EvidenceList chunks={evidence} />
              </div>
            </section>
          </section>

          <footer className="bottom-composer">
            <Composer
              question={question}
              setQuestion={setQuestion}
              mode={mode}
              setMode={setMode}
              onAsk={askQuestion}
              onUpload={() => fileInputRef.current?.click()}
              isAsking={isAsking}
              disabled={false}
            />
          </footer>
        </>
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
  setMode: (value: Mode) => void;
  onAsk: () => void;
  onUpload: () => void;
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
      <div className="mode-switch">
        <button className={props.mode === "text" ? "active" : ""} type="button" onClick={() => props.setMode("text")}>
          文本
        </button>
        <button className={props.mode === "visual" ? "active" : ""} type="button" onClick={() => props.setMode("visual")}>
          图像
        </button>
      </div>
      <button className="send-button" type="button" onClick={props.onAsk} disabled={props.isAsking}>
        {props.isAsking ? <Loader2 className="spin" size={20} /> : <Send size={20} />}
      </button>
    </div>
  );
}

function Panel(props: {
  active: boolean;
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <section className={props.active ? "panel active" : "panel"} onClick={props.onClick}>
      <header>
        <div className="panel-title">
          {props.icon}
          <h3>{props.title}</h3>
        </div>
        <span>{props.active ? "当前栏目" : "点击切换"}</span>
      </header>
      {props.active && props.children}
    </section>
  );
}

function SettingsBlock(props: { title: string; children: React.ReactNode }) {
  return (
    <details className="settings" open>
      <summary>
        <Settings2 size={18} />
        {props.title}
      </summary>
      <div className="settings-grid">{props.children}</div>
    </details>
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

function NumberRow(props: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <input
        type="number"
        value={props.value}
        onChange={(event) => props.onChange(Number(event.target.value))}
      />
    </label>
  );
}

function EvidenceList({ chunks }: { chunks: RetrievedChunk[] }) {
  if (!chunks.length) {
    return <p className="muted">暂无检索证据。</p>;
  }

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
