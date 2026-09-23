import os
import urllib.request
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import google.generativeai as genai

# -----------------------------
# 1. 폰트 깨짐 방지 (한글 폰트 설정)
# -----------------------------
@st.cache_resource
def load_korean_font():
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        urllib.request.urlretrieve(url, font_path)
    
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

load_korean_font()

# -----------------------------
# 2. Streamlit 기본 설정
# -----------------------------
st.set_page_config(page_title="PID 제어기 시뮬레이터", layout="wide")
st.title("🎛️ 온오프 / P / PI / PID 제어기 시뮬레이터")

# -----------------------------
# 3. [공정 모델 조건 고정]
# 300·dT/dt = -(T - 25) + 16·u(t-30), SP=1000℃, 총 5400초, dt=1초
# -----------------------------
TAU = 300.0
DELAY = 30.0
PROCESS_GAIN = 16.0
AMBIENT = 25.0
SP = 1000.0
TOTAL_TIME = 5400.0
DT = 1.0

# -----------------------------
# 4. 사이드바: PID 튜닝 파라미터 (조건 기본값: Kc=0.3, Ti=200, Td=25)
# -----------------------------
st.sidebar.header("🎛️ PID 튜닝 파라미터")
Kc = st.sidebar.number_input("비례 이득 (Kc)", value=0.3, step=0.05, format="%.2f")
Ti = st.sidebar.number_input("적분 시간 (τI, 초)", value=200.0, step=10.0, format="%.1f")
Td = st.sidebar.number_input("미분 시간 (τD, 초)", value=25.0, step=5.0, format="%.1f")

# -----------------------------
# 5. 제어기 시뮬레이션 함수들
# -----------------------------

# ① 온오프 제어 (히스테리시스 ±5℃: T <= 995 -> u=100, T >= 1005 -> u=0)
def simulate_onoff():
    n = int(TOTAL_TIME / DT)
    t = np.arange(n + 1) * DT
    temp = np.zeros(n + 1)
    output = np.zeros(n + 1)
    temp[0] = AMBIENT
    delay_steps = max(1, int(DELAY / DT))
    delay_buffer = [0.0] * delay_steps

    for k in range(n):
        T = temp[k]
        if T <= (SP - 5.0):
            u = 100.0
        elif T >= (SP + 5.0):
            u = 0.0
        else:
            u = output[k - 1] if k > 0 else 100.0

        output[k] = u
        delayed_u = delay_buffer.pop(0)
        delay_buffer.append(u)

        dTdt = (-(T - AMBIENT) + PROCESS_GAIN * delayed_u) / TAU
        temp[k + 1] = T + DT * dTdt

    output[-1] = output[-2]
    return t, temp, output

# ② P 제어: u = 0.3 * e (0~100% 제한)
def simulate_p():
    n = int(TOTAL_TIME / DT)
    t = np.arange(n + 1) * DT
    temp = np.zeros(n + 1)
    output = np.zeros(n + 1)
    temp[0] = AMBIENT
    delay_steps = max(1, int(DELAY / DT))
    delay_buffer = [0.0] * delay_steps

    for k in range(n):
        e = SP - temp[k]
        u = float(np.clip(0.3 * e, 0.0, 100.0))
        output[k] = u
        delayed_u = delay_buffer.pop(0)
        delay_buffer.append(u)

        dTdt = (-(temp[k] - AMBIENT) + PROCESS_GAIN * delayed_u) / TAU
        temp[k + 1] = temp[k] + DT * dTdt

    output[-1] = output[-2]
    return t, temp, output

# ③ PI 제어 (비교용)
def simulate_pi(Kc_val, Ti_val):
    n = int(TOTAL_TIME / DT)
    t = np.arange(n + 1) * DT
    temp = np.zeros(n + 1)
    output = np.zeros(n + 1)
    temp[0] = AMBIENT
    delay_steps = max(1, int(DELAY / DT))
    delay_buffer = [0.0] * delay_steps

    integral = 0.0

    for k in range(n):
        e = SP - temp[k]
        candidate_integral = integral + (e / Ti_val) * DT
        raw_output = Kc_val * (e + candidate_integral)
        u = float(np.clip(raw_output, 0.0, 100.0))

        # Anti-windup (포화 시 적분 중지)
        saturating_high = (raw_output > 100.0) and (e > 0)
        saturating_low = (raw_output < 0.0) and (e < 0)
        if not (saturating_high or saturating_low):
            integral = candidate_integral

        output[k] = u
        delayed_u = delay_buffer.pop(0)
        delay_buffer.append(u)

        dTdt = (-(temp[k] - AMBIENT) + PROCESS_GAIN * delayed_u) / TAU
        temp[k + 1] = temp[k] + DT * dTdt

    output[-1] = output[-2]
    return t, temp, output

# ④ PID 제어: 출력 포화 시 적분 누적 중지 (Anti-windup 적용)
def simulate_pid(Kc_val, Ti_val, Td_val):
    n = int(TOTAL_TIME / DT)
    t = np.arange(n + 1) * DT
    temp = np.zeros(n + 1)
    output = np.zeros(n + 1)
    temp[0] = AMBIENT
    delay_steps = max(1, int(DELAY / DT))
    delay_buffer = [0.0] * delay_steps

    integral = 0.0
    previous_error = SP - temp[0]

    for k in range(n):
        e = SP - temp[k]
        derivative = (e - previous_error) / DT
        candidate_integral = integral + (e / Ti_val) * DT

        raw_output = Kc_val * (e + candidate_integral + Td_val * derivative)
        u = float(np.clip(raw_output, 0.0, 100.0))

        # Anti-windup: 출력 포화 시 적분 누적 중지
        saturating_high = (raw_output > 100.0) and (e > 0)
        saturating_low = (raw_output < 0.0) and (e < 0)
        if not (saturating_high or saturating_low):
            integral = candidate_integral

        output[k] = u
        delayed_u = delay_buffer.pop(0)
        delay_buffer.append(u)

        dTdt = (-(temp[k] - AMBIENT) + PROCESS_GAIN * delayed_u) / TAU
        temp[k + 1] = temp[k] + DT * dTdt
        previous_error = e

    output[-1] = output[-2]
    return t, temp, output

# -----------------------------
# 6. 연산 실행 및 지표 계산
# -----------------------------
t, onoff_temp, onoff_output = simulate_onoff()
_, p_temp, _ = simulate_p()
_, pi_temp, _ = simulate_pi(Kc, Ti)
_, pid_temp, _ = simulate_pid(Kc, Ti, Td)

# 지표 계산
# 1) 온오프의 진동 폭 (마지막 30분 = 3600초 ~ 5400초 구간)
last_30min = t >= (TOTAL_TIME - 1800)
onoff_amp = np.max(onoff_temp[last_30min]) - np.min(onoff_temp[last_30min])

# 2) P 제어의 잔류편차 (마지막 5분 평균 기준)
last_5min = t >= (TOTAL_TIME - 300)
p_residual = SP - np.mean(p_temp[last_5min])

# 3) PID 제어의 최종 오차
pid_final_error = SP - pid_temp[-1]

# -----------------------------
# 7. 그래프 출력 (요구조건 3개)
# -----------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# 그래프 1) 온오프의 온도 전체 곡선
ax1.plot(t / 60, onoff_temp, label="ON/OFF 온도", color="tab:blue")
ax1.axhline(SP, color="red", linestyle="--", label="설정값 (1000°C)")
ax1.set_title("1) 온오프 제어 온도 전체 곡선")
ax1.set_ylabel("온도 (°C)")
ax1.set_xlabel("시간 (min)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# 그래프 2) 40~70분 구간 확대: 온도와 히터 출력(계단 모양)
mask_40_70 = (t >= 2400) & (t <= 4200)
ax2.plot(t[mask_40_70] / 60, onoff_temp[mask_40_70], label="온도", color="tab:blue")
ax2.axhline(SP, color="red", linestyle="--", label="설정값 (1000°C)")
ax2.set_title("2) 40~70분 구간 확대: 온도 + 히터 출력")
ax2.set_ylabel("온도 (°C)")
ax2.set_xlabel("시간 (min)")
ax2.grid(True, alpha=0.3)

ax2_twin = ax2.twinx()
ax2_twin.step(t[mask_40_70] / 60, onoff_output[mask_40_70], where="post", color="tab:orange", alpha=0.7, label="히터 출력 (%)")
ax2_twin.set_ylabel("히터 출력 (%)")
ax2_twin.set_ylim(-5, 105)

# 그래프 3) 세 제어기의 온도 비교 (y축 750~1100℃로 확대)
ax3.plot(t / 60, onoff_temp, label="ON/OFF", alpha=0.5)
ax3.plot(t / 60, p_temp, label="P 제어 (u=0.3e)", alpha=0.7)
ax3.plot(t / 60, pi_temp, label=f"PI 제어 (Kc={Kc}, τI={Ti}s)", alpha=0.7)
ax3.plot(t / 60, pid_temp, label=f"PID 제어 (Kc={Kc}, τI={Ti}s, τD={Td}s)", linewidth=2, color="tab:green")
ax3.axhline(SP, color="red", linestyle="--", label="설정값 (1000°C)")
ax3.set_ylim(750, 1100)  # y축 750~1100℃ 정확하게 고정
ax3.set_title("3) 제어기별 온도 비교 (Y축 확대: 750 ~ 1100°C)")
ax3.set_xlabel("시간 (min)")
ax3.set_ylabel("온도 (°C)")
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
st.pyplot(fig)

# -----------------------------
# 8. 마지막 결과 지표 출력
# -----------------------------
st.divider()
st.subheader("📊 제어 성능 측정 결과")

col1, col2, col3 = st.columns(3)
col1.metric("① 온오프 진동 폭 (마지막 30분)", f"{onoff_amp:.2f} °C")
col2.metric("② P 제어 잔류편차", f"{p_residual:.2f} °C")
col3.metric("③ PID 최종 오차", f"{pid_final_error:.2f} °C")

# -----------------------------
# 9. AI 분석 연동 (Streamlit Secrets)
# -----------------------------
st.divider()
if st.button("Gemini AI로 튜닝 분석받기"):
    if "GEMINI_API_KEY" in st.secrets:
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            제어공학 실습 평가를 진행해 주세요.
            - 공정: FOPDT (Tau=300, Delay=30, Kp=16, Ambient=25, SP=1000)
            - 설정된 PID 파라미터: Kc={Kc}, Ti={Ti}s, Td={Td}s
            - 결과: ON/OFF 진동폭={onoff_amp:.2f}°C, P 잔류편차={p_residual:.2f}°C, PID 최종오차={pid_final_error:.2f}°C

            위 결과를 바탕으로 P, PI, PID 제어기의 특성을 비교하고 현재 튜닝 파라미터에 대한 피드백을 간략히 작성해 주세요.
            """
            with st.spinner("AI 분석 중..."):
                res = model.generate_content(prompt)
                st.write(res.text)
        except Exception as e:
            st.error(f"오류 발생: {e}")
    else:
        st.warning("Secrets에 GEMINI_API_KEY를 등록해 주세요.")
