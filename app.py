from datetime import datetime, timedelta, timezone
import json
import os
from google import genai
from google.genai import types
import gspread
import numpy as np
import streamlit as st

st.set_page_config(page_title="KEC AI 규정 검색기", page_icon="⚡", layout="centered")
st.title("⚡ KEC(한국전기설비규정) AI 어시스턴트")
st.caption("1,233페이지 분량의 KEC 원문 및 표 데이터를 기반으로 조항과 기준 수치를 찾아드립니다.")

# 1. 배포 환경(Streamlit Secrets) 또는 환경변수에서 API 키 자동 로드
api_key = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY", "")

# 2. 사이드바 설정 (로컬 테스트 시 키 직접 입력 지원)
with st.sidebar:
    if not api_key:
        st.subheader("🔑 API 키 설정")
        api_key = st.text_input(
            "Gemini API 키를 입력하세요",
            type="password",
            placeholder="발급받은 API 키 붙여넣기",
            help="배포 시 Streamlit Cloud의 Secrets에 GEMINI_API_KEY를 등록하면 이 창은 자동으로 숨겨집니다.",
        )
        st.divider()

    st.header("⚙️ 답변 모드 설정")
    answer_mode = st.radio(
        "답변 방식을 선택하세요",
        ["⚡ 빠른 답변 (기본 · Flash-Lite)", "🔍 정밀 분석 (복합 조항 · Flash)"],
        index=0,
    )
    st.caption("일반 조항·수치 조회는 '빠른 답변'으로 1~2초 만에 출력되며, 여러 단서 조항 비교가 필요할 때 '정밀 분석'을 선택하세요.")
    st.divider()

    st.header("💡 사용 안내")
    st.markdown("- 실무 기준, 절연내력 시험전압, 접지, ESS 시설 기준 등을 질문해 보세요.\n- 답변 하단의 **[근거 KEC 원문 조항 보기]**를 누르면 실제 규정 조항과 표를 확인할 수 있습니다.")
    st.divider()

    st.subheader("자주 묻는 예시 질문")
    sample_questions = [
        "저압 전로 누설전류 기준 알려줘",
        "22.9kV 다중접지 전로 절연내력 시험전압은?",
        "전기저장장치(ESS) 제어 및 보호장치 기준은?",
    ]
    selected_sample = None
    for sq in sample_questions:
        if st.button(sq, use_container_width=True):
            selected_sample = sq

if not api_key:
    st.info("👈 왼쪽 사이드바에 Gemini API 키를 입력하면 바로 챗봇을 사용할 수 있습니다.")
    st.stop()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 3. 데이터, AI 클라이언트 및 구글 시트 로거 캐싱
@st.cache_resource
def load_resources(key: str):
    client = genai.Client(api_key=key)
    with open(os.path.join(BASE_DIR, "kec_chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    vectors = np.load(os.path.join(BASE_DIR, "kec_vectors.npz"))["vectors"]
    return client, chunks, vectors

@st.cache_resource(ttl=3600)
def get_log_worksheet():
    if "GCP_SERVICE_ACCOUNT_JSON" not in st.secrets:
        return None
    raw_secret = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
    creds_dict = json.loads(raw_secret) if isinstance(raw_secret, str) else dict(raw_secret)
    gc = gspread.service_account_from_dict(creds_dict)
    return gc.open("KEC_챗봇_사용기록").sheet1

def log_question(question: str, mode_label: str, answer_text: str):
    try:
        ws = get_log_worksheet()
        if ws is None:
            return
        kst = timezone(timedelta(hours=9))
        now_str = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
        summary = (answer_text or "").replace("\n", " ").strip()[:150]
        ws.append_row([now_str, mode_label, question, summary], value_input_option="USER_ENTERED")
    except Exception:
        pass  # 시트 연결 오류가 발생해도 사용자 답변 출력에는 영향을 주지 않음

@st.cache_data(ttl=3600)
def get_candidate_models(_client, prefer_lite: bool = True):
    if prefer_lite:
        base = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-3.8-flash"]
    else:
        base = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]

    candidates = []
    available = []
    try:
        for m in _client.models.list():
            name = m.name.replace("models/", "")
            if "gemini" in name and not any(
                x in name for x in ["embedding", "imagen", "veo", "tts", "audio", "live", "computer", "pro"]
            ):
                available.append(name)
    except Exception:
        return base

    if available:
        for b in base:
            if b in available and b not in candidates:
                candidates.append(b)
        others = [n for n in available if n not in candidates]
        if prefer_lite:
            others.sort(key=lambda x: (0 if "lite" in x else 1))
        else:
            others.sort(key=lambda x: (0 if "lite" not in x else 1))
        candidates.extend(others)

    return candidates if candidates else base

client, chunks, vectors = load_resources(api_key)

SYSTEM_PROMPT = """당신은 한국전기설비규정(KEC) 전문 AI 어시스턴트입니다.
반드시 아래 제공된 [참고 KEC 규정]의 내용과 표 수치만을 근거로 정확하게 답변하세요.
1. 답변할 때 근거가 된 KEC 조항 번호와 제목(예: KEC 132)을 반드시 명시하세요.
2. 표에 있는 수치, 전압, 단면적, 단서 조항(다만, ~인 경우)을 빠뜨리지 말고 정확히 안내하세요.
3. 제공된 규정 내용에 없는 사항은 추측해서 지어내지 말고, '제공된 KEC 규정 데이터에서 해당 내용을 찾을 수 없습니다'라고 답하세요."""

def retrieve_kec_sources(question: str, top_k: int = 3):
    q_res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    q_vec = np.array(q_res.embeddings[0].values, dtype=np.float32)
    q_vec = q_vec / np.maximum(np.linalg.norm(q_vec), 1e-12)

    scores = np.dot(vectors, q_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]

    context_list = []
    sources = []
    for idx in top_indices:
        c = chunks[idx]
        sources.append({"title": c["title"], "content": c["content"]})
        context_list.append(f"### 조항: {c['title']}\n{c['content']}")

    return sources, "\n\n".join(context_list)

def stream_kec_answer(question: str, context_text: str, prefer_lite: bool = True):
    prompt = f"[참고 KEC 규정]\n{context_text}\n\n[사용자 질문]\n{question}"
    last_err = None

    for model_name in get_candidate_models(client, prefer_lite=prefer_lite):
        try:
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                ),
            )
            iterator = iter(response_stream)
            first_chunk = next(iterator)
            if first_chunk.text:
                yield first_chunk.text
            for chunk in iterator:
                if chunk.text:
                    yield chunk.text
            return
        except StopIteration:
            return
        except Exception as e:
            last_err = e
            if any(code in str(e) for code in ["503", "UNAVAILABLE", "404", "NOT_FOUND", "429"]):
                continue
            raise e

    if last_err:
        raise last_err

# 4. 대화 기록 및 세션당 질문 횟수 제한
if "messages" not in st.session_state:
    st.session_state.messages = []
if "q_count" not in st.session_state:
    st.session_state.q_count = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📖 근거 KEC 원문 조항 보기"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['title']}**")
                    st.text(s["content"])
                    st.divider()

user_input = st.chat_input("KEC 규정에 대해 궁금한 점을 입력하세요...")
prompt_to_run = selected_sample or user_input

if prompt_to_run:
    if st.session_state.q_count >= 20:
        st.warning("베타 테스트 기간 중 1회 접속당 최대 20회까지 질문할 수 있습니다. 새로고침 후 다시 이용해 주세요.")
    else:
