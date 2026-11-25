// Behaviour test for app/static/js/main.js using a minimal stand-in for the browser (no external packages).
const fs = require("fs");
const path = require("path");
const assert = require("assert");

function makeEl(extra) {
  return Object.assign({ hidden: false, disabled: false, textContent: "", attrs: {}, listeners: {}, clicked: 0, value: "x",
    addEventListener(t, f) { this.listeners[t] = f; }, setAttribute(k, v) { this.attrs[k] = v; },
    removeAttribute(k) { delete this.attrs[k]; delete this[k]; }, getAttribute(k) { return this.attrs[k]; },
    click() { this.clicked++; } }, extra || {});
}
const tick = () => new Promise((r) => setTimeout(r, 0));

class FakeFile { constructor(parts, name, opts) { this.name = name; this.type = (opts || {}).type || ""; this.size = parts[0].size; } }
class FakeDT { constructor() { this.list = []; this.items = { add: (f) => this.list.push(f) }; } get files() { return this.list; } }

function run(opts) {
  opts = opts || {};
  const els = {
    "upload-form": makeEl(), image: makeEl({ files: [] }), preview: makeEl(), "preview-box": makeEl({ hidden: true }),
    "client-message": makeEl({ hidden: true }), analyse: makeEl(), busy: makeEl({ hidden: true }),
    camera: makeEl({ files: [] }), "camera-button": makeEl({ hidden: true }) };
  const revoked = [];
  let n = 0;
  const win = makeEl({ File: FakeFile, DataTransfer: FakeDT, navigator: { maxTouchPoints: opts.touch ? 5 : 0 } });
  if (opts.touch) { win.ontouchstart = null; }
  if (!opts.noResize) {
    win.createImageBitmap = () => Promise.resolve({ width: 4000, height: 3000, close() {} });
  }
  const canvas = { width: 0, height: 0, getContext: () => ({ drawImage() {} }),
    toBlob(cb) { cb(opts.blobFails ? null : { size: 800 * 1024 }); } };
  const body = makeEl({ attrs: { "data-max-mb": "5", "data-max-side": "1600" } });
  const doc = { getElementById: (id) => els[id], body, createElement: () => canvas };
  const URLstub = { createObjectURL: () => "blob:fake-" + (++n), revokeObjectURL: (u) => revoked.push(u) };
  const code = fs.readFileSync(path.join(__dirname, "../../app/static/js/main.js"), "utf8");
  new Function("document", "window", "URL", "parseInt", code)(doc, win, URLstub, parseInt);
  return { els, win, revoked, canvas };
}
const file = (name, type, size) => ({ name, type, size });
const choose = (t, f) => { t.els.image.files = f ? [f] : []; t.els.image.listeners.change(); };

(async () => {
  let t = run();
  assert.strictEqual(t.els.analyse.disabled, true, "button starts disabled when the script runs");
  assert.strictEqual(t.els["camera-button"].hidden, true, "camera button stays hidden on a computer");

  // unsupported type and oversize (no way to shrink) are explained at once
  t = run(); choose(t, file("notes.txt", "text/plain", 100)); await tick();
  assert.ok(/not supported/.test(t.els["client-message"].textContent)); assert.strictEqual(t.els.analyse.disabled, true);
  t = run({ noResize: true }); choose(t, file("big.jpg", "image/jpeg", 6 * 1024 * 1024)); await tick();
  assert.ok(/larger than 5 MB/.test(t.els["client-message"].textContent)); assert.strictEqual(t.els.analyse.disabled, true);

  // a small valid file is previewed and enabled without being changed
  t = run(); choose(t, file("leaf.JPG", "image/jpeg", 500 * 1024)); await tick();
  assert.strictEqual(t.els.preview.src, "blob:fake-1"); assert.strictEqual(t.els["preview-box"].hidden, false);
  assert.strictEqual(t.els.analyse.disabled, false); assert.strictEqual(t.els["client-message"].hidden, true);
  assert.strictEqual(t.els.image.files[0].name, "leaf.JPG");

  // a large phone photograph is shrunk to a JPEG and placed back in the form field
  t = run(); choose(t, file("IMG_001.png", "image/png", 6 * 1024 * 1024)); await tick();
  assert.strictEqual(t.els.analyse.disabled, false, "shrunk photograph accepted");
  assert.strictEqual(t.els.image.files[0].name, "IMG_001.jpg"); assert.strictEqual(t.els.image.files[0].type, "image/jpeg");
  assert.ok(t.els.image.files[0].size <= 5 * 1024 * 1024);
  assert.ok(t.canvas.width <= 1600 && t.canvas.height <= 1600 && t.canvas.width === 1600, "longest side limited to 1600");

  // if shrinking fails and the original is too large, the user is told
  t = run({ blobFails: true }); choose(t, file("IMG_002.jpg", "image/jpeg", 7 * 1024 * 1024)); await tick();
  assert.ok(/larger than 5 MB/.test(t.els["client-message"].textContent)); assert.strictEqual(t.els.analyse.disabled, true);

  // choosing another file releases the earlier preview address; clearing the choice disables the button
  t = run(); choose(t, file("a.jpg", "image/jpeg", 1000)); await tick();
  choose(t, file("b.png", "image/png", 1000)); await tick();
  assert.deepStrictEqual(t.revoked, ["blob:fake-1"]);
  choose(t, null); await tick(); assert.strictEqual(t.els.analyse.disabled, true); assert.strictEqual(t.els["preview-box"].hidden, true);

  // camera button appears on touch devices, opens the camera input, and its photograph goes into the form field
  t = run({ touch: true });
  assert.strictEqual(t.els["camera-button"].hidden, false);
  t.els["camera-button"].listeners.click(); assert.strictEqual(t.els.camera.clicked, 1);
  t.els.camera.files = [file("photo.jpg", "image/jpeg", 300 * 1024)]; t.els.camera.listeners.change(); await tick();
  assert.strictEqual(t.els.image.files[0].name, "photo.jpg"); assert.strictEqual(t.els.analyse.disabled, false);
  assert.strictEqual(t.els.camera.value, "", "camera field cleared so the same photograph can be taken again");

  // submit shows the loading indicator and blocks a second click; back-button restore clears it
  t = run(); choose(t, file("leaf.png", "image/png", 1000)); await tick();
  t.els["upload-form"].listeners.submit();
  assert.strictEqual(t.els.busy.hidden, false); assert.strictEqual(t.els.analyse.disabled, true);
  t.win.listeners.pageshow({ persisted: true }); assert.strictEqual(t.els.busy.hidden, true);

  console.log("frontend script behaviour: all checks passed");
})().catch((e) => { console.error(e); process.exit(1); });
