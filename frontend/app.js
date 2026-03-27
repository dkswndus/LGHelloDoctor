const BACKEND_URL = "http://localhost:8000";

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let lastAudioBase64 = null;

const recordBtn = document.getElementById("recordBtn");
const recordLabel = document.getElementById("recordLabel");
const recordStatus = document.getElementById("recordStatus");
const loadingSection = document.getElementById("loadingSection");
const resultSection = document.getElementById("resultSection");
const errorSection = document.getElementById("errorSection");
const audioPlayer = document.getElementById("audioPlayer");

recordBtn.addEventListener("click", toggleRecording);
document.getElementById("replayBtn").addEventListener("click", playAudio);

async function toggleRecording() {
  if (!isRecording) {
    await startRecording();
  } else {
    stopRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(audioChunks, { type: "audio/webm" });
      await sendAudio(blob);
    };

    mediaRecorder.start();
    isRecording = true;
    recordBtn.classList.add("recording");
    recordLabel.textContent = "중지";
    recordStatus.textContent = "녹음 중...";
    resetUI(false);
  } catch (e) {
    showError("마이크 접근 권한이 필요합니다.");
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  isRecording = false;
  recordBtn.classList.remove("recording");
  recordLabel.textContent = "말하기";
  recordStatus.textContent = "";
  showLoading();
}

async function sendAudio(blob) {
  let latitude = null;
  let longitude = null;
  try {
    const pos = await getLocation();
    latitude = pos.coords.latitude;
    longitude = pos.coords.longitude;
  } catch {
    // 위치 권한 없으면 병원 검색 생략
  }

  const formData = new FormData();
  formData.append("file", blob, "recording.webm");
  if (latitude !== null) formData.append("latitude", latitude);
  if (longitude !== null) formData.append("longitude", longitude);

  try {
    const res = await fetch(`${BACKEND_URL}/analyze-audio`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("서버 오류 (" + res.status + ")");
    const data = await res.json();
    showResult(data);
  } catch (e) {
    showError(e.message || "서버 연결에 실패했습니다.");
  }
}

function getLocation() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) return reject(new Error("위치 미지원"));
    navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 5000 });
  });
}

function showResult(data) {
  hideLoading();
  resultSection.classList.remove("hidden");
  errorSection.classList.add("hidden");

  document.getElementById("transcriptText").textContent = data.transcript || "-";
  document.getElementById("assistantText").textContent =
    data.assistant_message || (data.triage && data.triage.assistant_message) || "-";

  const hospital = data.triage && data.triage.hospital;
  const hospitalCard = document.getElementById("hospitalCard");
  if (hospital) {
    document.getElementById("hospitalName").textContent = hospital.name;
    document.getElementById("hospitalAddress").textContent =
      hospital.address ? "📍 " + hospital.address : "";
    document.getElementById("hospitalPhone").textContent =
      hospital.phone_number ? "📞 " + hospital.phone_number : "";
    hospitalCard.classList.remove("hidden");
  } else {
    hospitalCard.classList.add("hidden");
    const searchError = data.triage && data.triage.hospital_search_error;
    if (searchError) {
      document.getElementById("hospitalName").textContent = "병원 검색 실패: " + searchError;
      hospitalCard.classList.remove("hidden");
    }
  }

  if (data.tts_audio_base64) {
    lastAudioBase64 = data.tts_audio_base64;
    playAudio();
  }
}

function playAudio() {
  if (!lastAudioBase64) return;
  const byteStr = atob(lastAudioBase64);
  const buf = new Uint8Array(byteStr.length);
  for (let i = 0; i < byteStr.length; i++) buf[i] = byteStr.charCodeAt(i);
  const blob = new Blob([buf], { type: "audio/mpeg" });
  audioPlayer.src = URL.createObjectURL(blob);
  audioPlayer.play();
}

function showLoading() {
  loadingSection.classList.remove("hidden");
  resultSection.classList.add("hidden");
  errorSection.classList.add("hidden");
}

function hideLoading() {
  loadingSection.classList.add("hidden");
}

function showError(msg) {
  hideLoading();
  errorSection.classList.remove("hidden");
  resultSection.classList.add("hidden");
  document.getElementById("errorText").textContent = msg;
}

function resetUI(clearResults) {
  if (clearResults === undefined) clearResults = true;
  loadingSection.classList.add("hidden");
  errorSection.classList.add("hidden");
  if (clearResults) {
    resultSection.classList.add("hidden");
    lastAudioBase64 = null;
  }
}
