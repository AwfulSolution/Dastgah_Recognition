@echo off
cd /d %~dp0\..
call .venv\Scripts\activate
streamlit run Dastgah_Classifier_v2\app_v2.py
