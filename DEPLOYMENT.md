# Research deployment checklist

## 0. Before recruitment
- Get the approval/permission required by your instructor/institution for human-participant research.
- Freeze the product catalog, prompts, model/version, questionnaire and experiment flow before collecting real data.
- Do not collect names, emails, phone numbers, college IDs or other direct identifiers unless your approved protocol specifically requires them.

## 1. Create the response Google Sheet
1. Create a new Google Sheet.
2. Name it something like `AI Consumer Preference Study Responses`.
3. Copy the Apps Script from `google_apps_script/Code.gs` into Extensions → Apps Script.
4. Save the project.
5. Run `setup()` once and approve the requested Google permissions.
6. Back in the sheet, you should see `Responses` and `ChatLogs` tabs.

## 2. Deploy the Apps Script endpoint
1. In Apps Script select Deploy → New deployment.
2. Select Web app.
3. Execute as: Me.
4. Who has access: Anyone.
5. Deploy and copy the `/exec` URL.
6. Keep that URL private; it is a data-ingestion endpoint.

## 3. Test locally first
Use `.env`:

AI_PROVIDER=MOCK
STORAGE_MODE=LOCAL
TEST_GROUP=AI

Run:

`streamlit run app.py`

Complete one fake test response. Confirm `data/participants.csv` and `data/chat_logs.jsonl` appear.

## 4. Put the project on GitHub
Create a repository and upload the project files. Do NOT upload `.env`, API keys, response CSVs, or chat logs.

## 5. Deploy Streamlit Community Cloud
Use the repository, branch `main`, and main file `app.py`.

Add these secrets/environment values in the Streamlit deployment settings:

AI_PROVIDER = GEMINI
STORAGE_MODE = APPS_SCRIPT
TEST_GROUP =
GEMINI_API_KEY = your real key
GEMINI_MODEL = your verified current Gemini model ID
GOOGLE_APPS_SCRIPT_URL = your Apps Script /exec URL

Never commit the Gemini key to GitHub.

## 6. Test the public URL
- Test from phone.
- Test AI condition with TEST_GROUP=AI only while privately testing.
- Test control condition with TEST_GROUP=CONTROL only while privately testing.
- Remove TEST_GROUP before recruitment.
- Complete one full test and verify the row appears in Google Sheets.
- Verify the chat log appears in ChatLogs.

## 7. Freeze the study
After pilot changes are complete, do not change products, prompts, questionnaire wording, scoring, ranking logic or model version while collecting the main sample unless the protocol explicitly allows it.

## 8. Recruitment
Share only the public Streamlit URL. Use the approved consent/recruitment wording. A QR code can point to the same URL.

## 9. Data backup
After collection, export the Responses and ChatLogs sheets as CSV/XLSX and keep a dated read-only research backup.
