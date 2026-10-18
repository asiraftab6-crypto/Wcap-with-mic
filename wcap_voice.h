#pragma once

//
// wcap_voice.h - microphone voice clean-up
//
// Pure C, no dependencies, no allocations. Everything runs on the audio capture thread.
//
// Signal chain (all internally at 48 kHz mono):
//
//   mic -> resample to 48k -> high-pass 90 Hz (rumble / hum)
//       -> [NoiseSuppress] STFT spectral noise suppression (Wiener gain, decision-directed)
//       -> [NoiseSuppress] soft expander (removes leftover hiss between words)
//       -> [Enhance] slow AGC (even voice level)
//       -> [Enhance] EQ: cut mud ~250 Hz, presence boost ~3 kHz, air boost ~7 kHz
//       -> [Enhance] compressor (3:1)
//       -> user gain -> peak limiter
//       -> resample to loopback rate -> FIFO (popped by the loopback thread, mixed into the recording)
//

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define VOICE_RATE      48000
#define VOICE_N         512                   // STFT size
#define VOICE_HOP       256                   // 50% overlap
#define VOICE_BINS      (VOICE_N / 2 + 1)
#define VOICE_OUT_CAP   32768u                // FIFO capacity in samples, power of 2
#define VOICE_PI        3.14159265358979f

// tuning
#define VOICE_NS_OVERSUB     2.0f             // noise over-estimation factor (power)
#define VOICE_NS_FLOOR       0.07f            // maximum suppression, about -23 dB
#define VOICE_GATE_THR       0.0016f          // about -56 dBFS
#define VOICE_GATE_FLOOR     0.25f            // expander never attenuates more than -12 dB
#define VOICE_AGC_TARGET     0.10f            // about -20 dBFS RMS
#define VOICE_AGC_MIN        0.60f
#define VOICE_AGC_MAX        3.00f            // up to about +9.5 dB
#define VOICE_AGC_ACTIVE     0.0035f          // only adapt when the voice is above ~-49 dBFS
#define VOICE_COMP_THR       0.126f           // about -18 dBFS
#define VOICE_COMP_INV_RATIO 0.3333f          // 3:1
#define VOICE_COMP_MAKEUP    1.40f            // about +3 dB
#define VOICE_LIM_CEIL       0.95f

typedef struct
{
	float b0, b1, b2, a1, a2, z1, z2;
}
VoiceBiquad;

typedef struct
{
	bool  NoiseSuppress;
	bool  Enhance;
	float Gain;

	// input resampler (mic rate -> 48k)
	bool   InBypass;
	double InStep;
	double InPos;
	float  InPrev;

	// output resampler (48k -> loopback rate)
	bool     OutBypass;
	double   OutStep;
	double   OutPos;
	float    OutPrev;

	VoiceBiquad HighPass;
	VoiceBiquad EqMud;
	VoiceBiquad EqPresence;
	VoiceBiquad EqAir;

	// time-domain state
	float GateEnv, GateGain;
	float AgcMs, AgcGain;
	float CompEnv;
	float LimGain;

	// time-domain coefficients
	float CEnvAtk, CEnvRel;
	float CGateOpen, CGateClose;
	float CAgcMs, CAgcGain;
	float CCompAtk, CCompRel;
	float CLimRel;

	// STFT
	float    Window[VOICE_N];
	float    CosT[VOICE_N / 2];
	float    SinT[VOICE_N / 2];
	uint16_t BitRev[VOICE_N];
	float    Re[VOICE_N];
	float    Im[VOICE_N];
	float    InFrame[VOICE_N];
	float    Ola[VOICE_N];
	float    HopIn[VOICE_HOP];
	uint32_t HopFill;
	float    Ps[VOICE_BINS];
	float    Noise[VOICE_BINS];
	float    GainPrev[VOICE_BINS];
	float    GammaPrev[VOICE_BINS];
	bool     NoiseInit;

	// output FIFO (loopback sample rate)
	float    Out[VOICE_OUT_CAP];
	uint32_t OutR;
	uint32_t OutW;
	bool     Primed;
	uint32_t PrimeSamples;
	uint32_t MaxLatency;
}
VoiceProcessor;

//
// interface
//

// V must be zero-initialized memory of sizeof(VoiceProcessor)
static void     VoiceProcessor_Init(VoiceProcessor* V, uint32_t MicRate, uint32_t OutRate, bool NoiseSuppress, bool Enhance, float Gain);
// feed mono float samples at MicRate
static void     VoiceProcessor_Push(VoiceProcessor* V, const float* Mono, uint32_t Count);
// pull Count samples at OutRate, missing samples are zero-filled, returns number of real samples
static uint32_t VoiceProcessor_Pop(VoiceProcessor* V, float* Dst, uint32_t Count);

//
// implementation
//

static float Voice__Coef(float Seconds)
{
	return 1.0f - expf(-1.0f / (Seconds * (float)VOICE_RATE));
}

static void Voice__BiquadSet(VoiceBiquad* F, float b0, float b1, float b2, float a0, float a1, float a2)
{
	F->b0 = b0 / a0;
	F->b1 = b1 / a0;
	F->b2 = b2 / a0;
	F->a1 = a1 / a0;
	F->a2 = a2 / a0;
	F->z1 = 0.0f;
	F->z2 = 0.0f;
}

static float Voice__BiquadRun(VoiceBiquad* F, float X)
{
	float Y = F->b0 * X + F->z1;
	F->z1 = F->b1 * X - F->a1 * Y + F->z2;
	F->z2 = F->b2 * X - F->a2 * Y;
	return Y;
}

// RBJ audio EQ cookbook
static void Voice__HighPass(VoiceBiquad* F, float Freq, float Q)
{
	float W = 2.0f * VOICE_PI * Freq / (float)VOICE_RATE;
	float C = cosf(W);
	float Alpha = sinf(W) / (2.0f * Q);
	Voice__BiquadSet(F, (1.0f + C) * 0.5f, -(1.0f + C), (1.0f + C) * 0.5f, 1.0f + Alpha, -2.0f * C, 1.0f - Alpha);
}

static void Voice__Peaking(VoiceBiquad* F, float Freq, float Q, float Db)
{
	float A = powf(10.0f, Db / 40.0f);
	float W = 2.0f * VOICE_PI * Freq / (float)VOICE_RATE;
	float C = cosf(W);
	float Alpha = sinf(W) / (2.0f * Q);
	Voice__BiquadSet(F, 1.0f + Alpha * A, -2.0f * C, 1.0f - Alpha * A, 1.0f + Alpha / A, -2.0f * C, 1.0f - Alpha / A);
}

static void Voice__HighShelf(VoiceBiquad* F, float Freq, float Db)
{
	float A = powf(10.0f, Db / 40.0f);
	float W = 2.0f * VOICE_PI * Freq / (float)VOICE_RATE;
	float C = cosf(W);
	float Alpha = sinf(W) * 0.5f * 1.41421356f;
	float T = 2.0f * sqrtf(A) * Alpha;
	Voice__BiquadSet(F,
		A * ((A + 1.0f) + (A - 1.0f) * C + T),
		-2.0f * A * ((A - 1.0f) + (A + 1.0f) * C),
		A * ((A + 1.0f) + (A - 1.0f) * C - T),
		(A + 1.0f) - (A - 1.0f) * C + T,
		2.0f * ((A - 1.0f) - (A + 1.0f) * C),
		(A + 1.0f) - (A - 1.0f) * C - T);
}

static void Voice__Fft(VoiceProcessor* V, float* Re, float* Im, bool Inverse)
{
	for (int i = 0; i < VOICE_N; i++)
	{
		int j = V->BitRev[i];
		if (j > i)
		{
			float T;
			T = Re[i]; Re[i] = Re[j]; Re[j] = T;
			T = Im[i]; Im[i] = Im[j]; Im[j] = T;
		}
	}

	for (int Len = 2; Len <= VOICE_N; Len <<= 1)
	{
		int Half = Len / 2;
		int Step = VOICE_N / Len;
		for (int i = 0; i < VOICE_N; i += Len)
		{
			for (int j = 0; j < Half; j++)
			{
				float Wr = V->CosT[j * Step];
				float Wi = Inverse ? V->SinT[j * Step] : -V->SinT[j * Step];
				float Ur = Re[i + j];
				float Ui = Im[i + j];
				float Xr = Re[i + j + Half];
				float Xi = Im[i + j + Half];
				float Tr = Xr * Wr - Xi * Wi;
				float Ti = Xr * Wi + Xi * Wr;
				Re[i + j]        = Ur + Tr;
				Im[i + j]        = Ui + Ti;
				Re[i + j + Half] = Ur - Tr;
				Im[i + j + Half] = Ui - Ti;
			}
		}
	}

	if (Inverse)
	{
		const float Scale = 1.0f / (float)VOICE_N;
		for (int i = 0; i < VOICE_N; i++)
		{
			Re[i] *= Scale;
			Im[i] *= Scale;
		}
	}
}

static void Voice__Emit(VoiceProcessor* V, float X)
{
	if (V->OutW - V->OutR >= VOICE_OUT_CAP)
	{
		V->OutR = V->OutW - VOICE_OUT_CAP + 1;
	}
	V->Out[V->OutW & (VOICE_OUT_CAP - 1)] = X;
	V->OutW++;
}

// final stage, one sample at 48 kHz
static void Voice__Post(VoiceProcessor* V, float X)
{
	// level follower (used by expander and AGC)
	float Ax = fabsf(X);
	V->GateEnv += (Ax > V->GateEnv ? V->CEnvAtk : V->CEnvRel) * (Ax - V->GateEnv);

	if (V->NoiseSuppress)
	{
		// soft downward expander for leftover noise between words
		float Target = 1.0f;
		if (V->GateEnv < VOICE_GATE_THR)
		{
			Target = V->GateEnv / VOICE_GATE_THR;
			if (Target < VOICE_GATE_FLOOR)
			{
				Target = VOICE_GATE_FLOOR;
			}
		}
		V->GateGain += (Target > V->GateGain ? V->CGateOpen : V->CGateClose) * (Target - V->GateGain);
		X *= V->GateGain;
	}

	if (V->Enhance)
	{
		// slow AGC, adapts only while somebody is talking
		if (V->GateEnv > VOICE_AGC_ACTIVE)
		{
			V->AgcMs += V->CAgcMs * (X * X - V->AgcMs);
			float Want = VOICE_AGC_TARGET / sqrtf(V->AgcMs + 1e-10f);
			if (Want < VOICE_AGC_MIN) Want = VOICE_AGC_MIN;
			if (Want > VOICE_AGC_MAX) Want = VOICE_AGC_MAX;
			V->AgcGain += V->CAgcGain * (Want - V->AgcGain);
		}
		X *= V->AgcGain;

		// voice EQ
		X = Voice__BiquadRun(&V->EqMud, X);
		X = Voice__BiquadRun(&V->EqPresence, X);
		X = Voice__BiquadRun(&V->EqAir, X);

		// compressor
		float Ac = fabsf(X);
		V->CompEnv += (Ac > V->CompEnv ? V->CCompAtk : V->CCompRel) * (Ac - V->CompEnv);
		float Gc = 1.0f;
		if (V->CompEnv > VOICE_COMP_THR)
		{
			Gc = powf(V->CompEnv / VOICE_COMP_THR, VOICE_COMP_INV_RATIO - 1.0f);
		}
		X *= Gc * VOICE_COMP_MAKEUP;
	}

	X *= V->Gain;

	// peak limiter
	float Need = 1.0f;
	float Ap = fabsf(X);
	if (Ap > VOICE_LIM_CEIL)
	{
		Need = VOICE_LIM_CEIL / Ap;
	}
	if (Need < V->LimGain)
	{
		V->LimGain = Need;
	}
	else
	{
		V->LimGain += V->CLimRel * (1.0f - V->LimGain);
		if (V->LimGain > Need)
		{
			V->LimGain = Need;
		}
	}
	X *= V->LimGain;

	// resample to output rate
	if (V->OutBypass)
	{
		Voice__Emit(V, X);
	}
	else
	{
		while (V->OutPos < 1.0)
		{
			Voice__Emit(V, V->OutPrev + (X - V->OutPrev) * (float)V->OutPos);
			V->OutPos += V->OutStep;
		}
		V->OutPos -= 1.0;
		V->OutPrev = X;
	}
}

// one STFT frame of noise suppression, produces VOICE_HOP output samples
static void Voice__Frame(VoiceProcessor* V)
{
	float* Re = V->Re;
	float* Im = V->Im;

	for (int n = 0; n < VOICE_N; n++)
	{
		Re[n] = V->InFrame[n] * V->Window[n];
		Im[n] = 0.0f;
	}
	Voice__Fft(V, Re, Im, false);

	float P[VOICE_BINS];
	for (int k = 0; k < VOICE_BINS; k++)
	{
		P[k] = Re[k] * Re[k] + Im[k] * Im[k];
	}

	if (!V->NoiseInit)
	{
		for (int k = 0; k < VOICE_BINS; k++)
		{
			V->Ps[k] = P[k];
			V->Noise[k] = P[k];
			V->GainPrev[k] = 1.0f;
			V->GammaPrev[k] = 1.0f;
		}
		V->NoiseInit = true;
	}

	// noise floor tracking: follows minimum of smoothed power, rises slowly
	float SumP = 0.0f;
	float SumN = 0.0f;
	for (int k = 0; k < VOICE_BINS; k++)
	{
		V->Ps[k] = 0.7f * V->Ps[k] + 0.3f * P[k];
		if (k >= 3 && k < 100) // speech band, roughly 280 Hz .. 9 kHz (bins are 93.75 Hz wide)
		{
			SumP += V->Ps[k];
			SumN += V->Noise[k];
		}
	}
	bool Speech = SumP > 3.0f * SumN;
	float Rise = Speech ? 1.0005f : 1.0040f;

	for (int k = 0; k < VOICE_BINS; k++)
	{
		if (V->Ps[k] < V->Noise[k])
		{
			V->Noise[k] += 0.25f * (V->Ps[k] - V->Noise[k]);
		}
		else
		{
			float Up = V->Noise[k] * Rise;
			V->Noise[k] = Up < V->Ps[k] ? Up : V->Ps[k];
		}
	}

	// Wiener gain with decision-directed a-priori SNR
	float G[VOICE_BINS];
	for (int k = 0; k < VOICE_BINS; k++)
	{
		float Nk = V->Noise[k] * VOICE_NS_OVERSUB + 1e-12f;
		float Gamma = P[k] / Nk;
		float Post = Gamma - 1.0f;
		if (Post < 0.0f) Post = 0.0f;
		float Xi = 0.96f * V->GainPrev[k] * V->GainPrev[k] * V->GammaPrev[k] + 0.04f * Post;
		float Gk = Xi / (1.0f + Xi);
		if (Gk < VOICE_NS_FLOOR) Gk = VOICE_NS_FLOOR;
		G[k] = Gk;
		V->GammaPrev[k] = Gamma;
	}

	// smooth across frequency to avoid "musical noise"
	float Gs[VOICE_BINS];
	Gs[0] = G[0];
	Gs[VOICE_BINS - 1] = G[VOICE_BINS - 1];
	for (int k = 1; k < VOICE_BINS - 1; k++)
	{
		Gs[k] = 0.25f * G[k - 1] + 0.5f * G[k] + 0.25f * G[k + 1];
	}

	for (int k = 0; k < VOICE_BINS; k++)
	{
		V->GainPrev[k] = Gs[k];
		Re[k] *= Gs[k];
		Im[k] *= Gs[k];
		if (k > 0 && k < VOICE_N / 2)
		{
			Re[VOICE_N - k] *= Gs[k];
			Im[VOICE_N - k] *= Gs[k];
		}
	}

	Voice__Fft(V, Re, Im, true);

	for (int n = 0; n < VOICE_N; n++)
	{
		V->Ola[n] += Re[n] * V->Window[n];
	}

	for (int n = 0; n < VOICE_HOP; n++)
	{
		Voice__Post(V, V->Ola[n]);
	}
	memmove(V->Ola, V->Ola + VOICE_HOP, (VOICE_N - VOICE_HOP) * sizeof(float));
	memset(V->Ola + (VOICE_N - VOICE_HOP), 0, VOICE_HOP * sizeof(float));
}

// one sample at 48 kHz
static void Voice__Sample(VoiceProcessor* V, float X)
{
	X = Voice__BiquadRun(&V->HighPass, X + 1e-20f);

	if (!V->NoiseSuppress)
	{
		Voice__Post(V, X);
		return;
	}

	V->HopIn[V->HopFill++] = X;
	if (V->HopFill == VOICE_HOP)
	{
		V->HopFill = 0;
		memmove(V->InFrame, V->InFrame + VOICE_HOP, (VOICE_N - VOICE_HOP) * sizeof(float));
		memcpy(V->InFrame + (VOICE_N - VOICE_HOP), V->HopIn, VOICE_HOP * sizeof(float));
		Voice__Frame(V);
	}
}

void VoiceProcessor_Init(VoiceProcessor* V, uint32_t MicRate, uint32_t OutRate, bool NoiseSuppress, bool Enhance, float Gain)
{
	memset(V, 0, sizeof(*V));

	V->NoiseSuppress = NoiseSuppress;
	V->Enhance = Enhance;
	V->Gain = Gain;

	V->InBypass = (MicRate == VOICE_RATE);
	V->InStep = (double)MicRate / (double)VOICE_RATE;
	V->InPos = 0.0;

	V->OutBypass = (OutRate == VOICE_RATE);
	V->OutStep = (double)VOICE_RATE / (double)OutRate;
	V->OutPos = 0.0;

	// STFT tables
	for (int i = 0; i < VOICE_N; i++)
	{
		// sqrt of periodic Hann: analysis * synthesis window sums to 1 at 50% overlap
		V->Window[i] = sqrtf(0.5f - 0.5f * cosf(2.0f * VOICE_PI * (float)i / (float)VOICE_N));

		int Rev = 0;
		for (int T = i, Len = VOICE_N; Len > 1; Len >>= 1)
		{
			Rev = (Rev << 1) | (T & 1);
			T >>= 1;
		}
		V->BitRev[i] = (uint16_t)Rev;
	}
	for (int i = 0; i < VOICE_N / 2; i++)
	{
		V->CosT[i] = cosf(2.0f * VOICE_PI * (float)i / (float)VOICE_N);
		V->SinT[i] = sinf(2.0f * VOICE_PI * (float)i / (float)VOICE_N);
	}

	// filters
	Voice__HighPass(&V->HighPass, 90.0f, 0.7071f);
	Voice__Peaking(&V->EqMud, 250.0f, 0.9f, -2.0f);
	Voice__Peaking(&V->EqPresence, 3000.0f, 0.9f, 2.5f);
	Voice__HighShelf(&V->EqAir, 7000.0f, 2.0f);

	// time constants
	V->CEnvAtk    = Voice__Coef(0.002f);
	V->CEnvRel    = Voice__Coef(0.060f);
	V->CGateOpen  = Voice__Coef(0.004f);
	V->CGateClose = Voice__Coef(0.120f);
	V->CAgcMs     = Voice__Coef(0.400f);
	V->CAgcGain   = Voice__Coef(0.350f);
	V->CCompAtk   = Voice__Coef(0.008f);
	V->CCompRel   = Voice__Coef(0.150f);
	V->CLimRel    = Voice__Coef(0.050f);

	V->GateGain = 1.0f;
	V->AgcGain  = 1.0f;
	V->AgcMs    = VOICE_AGC_TARGET * VOICE_AGC_TARGET;
	V->LimGain  = 1.0f;

	V->PrimeSamples = OutRate * 15u / 1000u;
	V->MaxLatency   = OutRate * 120u / 1000u;
}

void VoiceProcessor_Push(VoiceProcessor* V, const float* Mono, uint32_t Count)
{
	for (uint32_t i = 0; i < Count; i++)
	{
		float X = Mono[i];
		if (!(X > -4.0f && X < 4.0f)) // NaN / garbage guard
		{
			X = 0.0f;
		}

		if (V->InBypass)
		{
			Voice__Sample(V, X);
		}
		else
		{
			while (V->InPos < 1.0)
			{
				Voice__Sample(V, V->InPrev + (X - V->InPrev) * (float)V->InPos);
				V->InPos += V->InStep;
			}
			V->InPos -= 1.0;
			V->InPrev = X;
		}
	}
}

uint32_t VoiceProcessor_Pop(VoiceProcessor* V, float* Dst, uint32_t Count)
{
	uint32_t Avail = V->OutW - V->OutR;

	// too much queued: skip ahead to keep voice in sync with video
	if (Avail > Count + V->MaxLatency)
	{
		V->OutR = V->OutW - (Count + V->PrimeSamples);
		Avail = Count + V->PrimeSamples;
	}

	if (!V->Primed)
	{
		if (Avail >= Count + V->PrimeSamples)
		{
			V->Primed = true;
		}
		else
		{
			memset(Dst, 0, Count * sizeof(float));
			return 0;
		}
	}

	uint32_t N = Avail < Count ? Avail : Count;
	for (uint32_t i = 0; i < N; i++)
	{
		Dst[i] = V->Out[(V->OutR + i) & (VOICE_OUT_CAP - 1)];
	}
	V->OutR += N;
	if (N < Count)
	{
		memset(Dst + N, 0, (Count - N) * sizeof(float));
		V->Primed = false; // underrun, build up a little cushion again
	}
	return N;
}
