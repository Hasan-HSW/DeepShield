# run.ps1 - تشغيل سريع لمشروع DeepShield على ويندوز
# طريقة الاستخدام: افتح PowerShell داخل مجلد المشروع ثم نفّذ:
#   .\run.ps1

Write-Host "== DeepShield - إعداد وتشغيل المشروع ==" -ForegroundColor Cyan

if (-Not (Test-Path "venv")) {
    Write-Host "إنشاء بيئة افتراضية جديدة..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "تفعيل البيئة الافتراضية..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

Write-Host "تثبيت المكتبات المطلوبة..." -ForegroundColor Yellow
pip install -r requirements.txt

if (-Not (Test-Path "models\model.keras")) {
    Write-Host "تدريب النموذج على جهازك (مرة واحدة فقط، يأخذ ثوانٍ)..." -ForegroundColor Yellow
    python -m src.train
}

Write-Host "تشغيل التطبيق على http://localhost:5000 ..." -ForegroundColor Green
python app.py
