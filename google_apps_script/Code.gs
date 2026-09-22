const RESPONSE_SHEET = 'Responses';
const CHAT_SHEET = 'ChatLogs';

function setup() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  ensureSheet_(ss, RESPONSE_SHEET);
  ensureSheet_(ss, CHAT_SHEET);
}

function ensureSheet_(ss, name) {
  let sheet = ss.getSheetByName(name);
  if (!sheet) sheet = ss.insertSheet(name);
  return sheet;
}

function doGet() {
  return json_({ok: true, service: 'AI Consumer Preference Study', status: 'ready'});
}

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents || '{}');
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    if (body.type === 'response') {
      appendRecord_(ensureSheet_(ss, RESPONSE_SHEET), body.record || {});
    } else if (body.type === 'chat') {
      appendRecord_(ensureSheet_(ss, CHAT_SHEET), body.record || {});
    } else {
      throw new Error('Unknown payload type');
    }
    return json_({ok: true});
  } catch (err) {
    return json_({ok: false, error: String(err)});
  }
}

function appendRecord_(sheet, record) {
  const flat = flatten_(record);
  const headers = sheet.getLastColumn() ? sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0] : [];
  const keys = Object.keys(flat);
  if (headers.length === 0 || (headers.length === 1 && headers[0] === '')) {
    sheet.getRange(1, 1, 1, keys.length).setValues([keys]);
    sheet.appendRow(keys.map(k => flat[k]));
    return;
  }
  const newHeaders = headers.slice();
  keys.forEach(k => { if (!newHeaders.includes(k)) newHeaders.push(k); });
  if (newHeaders.length !== headers.length) {
    sheet.getRange(1, 1, 1, newHeaders.length).setValues([newHeaders]);
  }
  sheet.appendRow(newHeaders.map(k => flat[k] === undefined ? '' : flat[k]));
}

function flatten_(obj, prefix, out) {
  prefix = prefix || '';
  out = out || {};
  Object.keys(obj).forEach(key => {
    const value = obj[key];
    const name = prefix ? prefix + '_' + key : key;
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      flatten_(value, name, out);
    } else if (Array.isArray(value)) {
      out[name] = JSON.stringify(value);
    } else {
      out[name] = value;
    }
  });
  return out;
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
