# Transient Detection System

A robust audio transient detection system using Combined Wavelet Transform (CWT) and time-domain envelope analysis with automatic parameter adaptation.

## Overview

This system detects transient events (attacks, onsets) in audio signals by combining:
- **Frequency-domain analysis** via Continuous Wavelet Transform (CWT)
- **Time-domain analysis** via envelope following
- **Automatic parameter detection** based on audio characteristics

## Features

- ✅ Dual-method detection (CWT + Time-domain) with intelligent combination
- ✅ Automatic audio analysis and parameter selection
- ✅ Batch processing for multiple files
- ✅ Comprehensive visualization
- ✅ Handles mono and stereo audio
- ✅ Optimized for different audio types (percussive, plucked, sustained)


<img width="2384" height="1484" alt="C_C5Yangqin_01_541_detection" src="https://github.com/user-attachments/assets/e2d8fa69-80dd-438d-91df-426ead3e6bb7" />
<img width="2383" height="1484" alt="castanets_192kbps_detection" src="https://github.com/user-attachments/assets/0447c4c0-70db-498e-9d43-2e0fcc1f12a1" />
<img width="2384" height="1484" alt="CHIME Kick 2017_detection" src="https://github.com/user-attachments/assets/e8ec9b95-4a30-46f5-8730-a70466505dbc" />
<img width="2384" height="1484" alt="glockenspiel_192kbps_detection" src="https://github.com/user-attachments/assets/d2f32764-7286-474d-b248-1bc2ef755d1a" />
<img width="2384" height="1484" alt="Piano_Hard_C4_detection" src="https://github.com/user-attachments/assets/92a6a633-1444-478e-81eb-9ae83029f083" />
<img width="2384" height="1484" alt="String_Pizzicato1_A#_detection" src="https://github.com/user-attachments/assets/60aab9d5-f8ac-493e-93b6-f57577f8806d" />
<img width="2384" height="1482" alt="ringnoord_detection" src="https://github.com/user-attachments/assets/1127e95e-086e-48ef-b927-cfd30580ba1f" />

### Requirements

```bash
pip install numpy scipy librosa matplotlib pycwt


