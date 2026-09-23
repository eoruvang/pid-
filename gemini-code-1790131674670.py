import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai

# -----------------------------
# Streamlit 기본 설정
# -----------------------------
st.set_page_config(page_title="PID 제어 시뮬레이터 + AI 분석", layout="wide")

st.title("🎛️ 온오프 / P / PID 제어 시뮬레이터 & AI 진단")
st.markdown("공정 변수와 PID 설정을 실시간으로 조절하고, Google Gemini AI를 통해 제어 성능을 분석받으세요.")

# -----------------------------
# 사이드바: 파라미터 설정
# -----------------------------
st.sidebar.header("⚙️ 1. 공정 모델 설정")
TAU = st.sidebar.number_input("시상수 (tau, 초)", value=300.0, step=10.0)
DELAY = st.sidebar.number_input("시간 지연 (delay, 초)", value=30.0, step=5.0)
PROCESS_GAIN = st.sidebar.number_input("공정 이득 (K_p)", value=16.0, step=1.0)
AMBIENT = st.sidebar.number_input("주위/초기 온도 (°C)", value=25.0, step=5.0)
SP = st.sidebar.number_input("목표 설정값 (SP, °C)", value=1000.0, step=50.0)
TOTAL_TIME = st.sidebar.number_input("총 시뮬레이션 시간 (초)", value=5400.0, step=600.0)
DT = 1.0

st.sidebar.header("🎛️ 2. PID 튜닝 파라미터")
Kc = st.sidebar.slider("Kc (비례 이득)", min_value=0.01, max_value=3.0, value=0.3, step=0.01)
Ti = st.sidebar.slider("Ti (적분 시간, 초)", min_value=1.0, max_value=1000.0, value=200.0, step=10.0)
Td = st.sidebar.slider("Td (미분 시간, 초)", min_value=0.0, max_value=200.0, value=25.0, step=5.0)

# -----------------------------
# 시뮬레이션 연산 함수
# -----------------------------
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
        if T <= SP - 5.0:
            u = 100.0
        elif T >= SP + 5.0:
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

def simulate_p():
    n = int(TOTAL_TIME / DT)
    t = np.arange(n + 1) * DT
    temp = np.zeros(n + 1)
    output = np.zeros(n + 1)
    temp[0] = AMBIENT
    delay_steps = max(1, int(DELAY / DT))
    delay_buffer = [0.0] * delay_steps

    for k in range(n):
        error = SP - temp[k]
        u = float(np.clip(0.3 * error, 0.0, 100.0))
        output[k] = u
        delayed_u = delay_buffer.pop(0)
        delay_buffer.append(u)

        dTdt = (-(temp[k] - AMBIENT) + PROCESS_GAIN * delayed_u) / TAU
        temp[k + 1] = temp[k] + DT * dTdt

    output[-1] = output[-2]
    return t, temp, output

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
        error = SP - temp[k]
        derivative = (error - previous_error) / DT
        candidate_integral = integral + (error / Ti_val) * DT

        raw_output = Kc_val * (error + candidate_integral + Td_val * derivative)
        u = float(np.clip(raw_output, 0.0, 100.0))

        saturating_high = (raw_output > 100.0) and (error > 0)
        saturating_low = (raw_output < 0.0) and (error < 0)

        if not (saturating_high or saturating_low):
            integral = candidate_integral

        u = float(np.clip(Kc_val * (error + integral + Td_val * derivative), 0.0, 100.0))
        output[k] = u

        delayed_u = delay_buffer.pop(0)
        delay_buffer.append(u)

        dTdt = (-(temp[k] - AMBIENT) + PROCESS_GAIN * delayed_u) / TAU
        temp[k + 1] = temp[k] + DT * dTdt
        previous_error = error

    output[-1] = output[-2]
    return t, temp, output

# 시뮬레이션 실행
t, onoff_temp, onoff_output = simulate_onoff()
_, p_temp, _ = simulate_p()
_, pid_temp, _ = simulate_pid(Kc, Ti, Td)

# 결과 지표 계산
last5 = t >= (TOTAL_TIME - 300)
onoff_amp = np.max(onoff_temp[last5]) - np.min(onoff_temp[last5])
p_final_avg = np.mean(p_temp[last5])
p_residual = SP - p_final_avg
pid_final_error = SP - pid_temp[-1]

# -----------------------------
# 결과 지표 출력
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("① ON/OFF 진동폭", f"{onoff_amp:.2f} °C")
col2.metric("② P 잔류편차", f"{p_residual:.2f} °C")
col3.metric("③ PID 최종오차", f"{pid_final_error:.2f} °C (최종: {pid_temp[-1]:.1f}°C)")

st.divider()

# -----------------------------
# 그래프 생성
# -----------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10))

ax1.plot(t / 60, onoff_temp, label="ON/OFF 온도", color="tab:blue")
ax1.axhline(SP, color="red", linestyle="--", label="설정값")
ax1.set_title("① ON/OFF 제어 온도 전체 곡선")
ax1.set_ylabel("온도 (°C)")
ax1.grid(True, alpha=0.3)
ax1.legend()

mask = (t >= 2400) & (t <= 4200)
ax2.plot(t[mask] / 60, onoff_temp[mask], label="온도", color="tab:blue")
ax2.axhline(SP, color="red", linestyle="--", label="설정값")
ax2.set_title("② 40~70분 구간 확대: 온도 + 히터 출력")
ax2.set_ylabel("온도 (°C)")
ax2.grid(True, alpha=0.3)

ax2b = ax2.twinx()
ax2b.step(t[mask] / 60, onoff_output[mask], where="post", color="tab:orange", alpha=0.7, label="히터 출력")
ax2b.set_ylabel("출력 (%)")
ax2b.set_ylim(-5, 105)

ax3.plot(t / 60, onoff_temp, label="ON/OFF", alpha=0.6)
ax3.plot(t / 60, p_temp, label="P 제어", alpha=0.6)
ax3.plot(t / 60, pid_temp, label=f"PID (Kc={Kc}, Ti={Ti}s, Td={Td}s)", linewidth=2, color="tab:green")
ax3.axhline(SP, color="red", linestyle="--", label="설정값")
ax3.set_ylim(SP - 250, SP + 100)
ax3.set_title("③ 제어기별 온도 비교")
ax3.set_xlabel("시간 (min)")
ax3.set_ylabel("온도 (°C)")
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
st.pyplot(fig)

# -----------------------------
# AI 분석 연동 (Streamlit Secrets 사용)
# -----------------------------
st.divider()
st.subheader("🤖 AI 제어 성능 분석")

if st.button("Gemini AI로 튜닝 상태 분석하기"):
    # Streamlit Cloud에 저장된 API 키 가져오기
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            당신은 제어공학 전문가입니다. 아래 PID 제어 시뮬레이션 결과를 분석하고 튜닝 피드백을 제공해 주세요.

            - 설정값(SP): {SP} °C
            - 공정 조건: 시상수={TAU}s, 지연시간={DELAY}s, 공정이득={PROCESS_GAIN}
            - 현재 PID 파라미터: Kc={Kc}, Ti={Ti}s, Td={Td}s
            - 결과: 최종 온도={pid_temp[-1]:.2f} °C, 최종 오차={pid_final_error:.2f} °C

            위 설정이 적절한지 평가하고, 응답 속도를 개선하거나 오버슈트/잔류편차를 줄이기 위해 Kc, Ti, Td를 어떻게 수정해야 할지 친절하게 알려주세요.
            """
            
            with st.spinner("AI가 시뮬레이션 결과를 분석 중입니다..."):
                response = model.generate_content(prompt)
                st.write(response.text)
        except Exception as e:
            st.error(f"AI 연동 오류: {e}")
    else:
        st.warning("Streamlit Cloud 설정(Secrets)에 GEMINI_API_KEY가 등록되지 않았습니다.")