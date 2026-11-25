/* Light enhancements only: preview, early advice on file type and size, shrinking large photographs,
   a camera button on touch devices, and a loading indicator. The server repeats every check, so the
   page also works without this script (the camera button then simply does not appear). */
(function () {
  "use strict";
  var form = document.getElementById("upload-form");
  var input = document.getElementById("image");
  if (!form || !input) { return; }

  var body = document.body;
  var maxMb = parseInt(body.getAttribute("data-max-mb"), 10) || 5;
  var maxSide = parseInt(body.getAttribute("data-max-side"), 10) || 1600;
  var maxBytes = maxMb * 1024 * 1024;
  var resizeAbove = 1024 * 1024;               // photographs larger than 1 MB are shrunk before upload
  var preview = document.getElementById("preview");
  var previewBox = document.getElementById("preview-box");
  var message = document.getElementById("client-message");
  var button = document.getElementById("analyse");
  var busy = document.getElementById("busy");
  var camera = document.getElementById("camera");
  var cameraButton = document.getElementById("camera-button");
  var replaceButton = document.getElementById("replace-photo");
  var allowedName = /\.(jpe?g|png)$/i;
  var allowedType = ["image/jpeg", "image/png"];
  var objectUrl = null;
  var token = 0;                                // ignores the result of an outdated choice

  function showMessage(text) { message.textContent = text; message.hidden = false; }

  function clearPreview() {
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    preview.removeAttribute("src");
    previewBox.hidden = true;
    if (replaceButton) { replaceButton.hidden = true; }
    message.textContent = "";
    message.hidden = true;
  }

  function reset() { clearPreview(); busy.hidden = true; button.disabled = !(input.files && input.files.length); }

  function jpgName(name) { return name.replace(/\.[^.]+$/, "") + ".jpg"; }

  /* Resolves with a smaller JPEG copy when that helps, otherwise with the original file. */
  function shrinkIfNeeded(file) {
    return new Promise(function (resolve) {
      var w = window;
      if (file.size <= resizeAbove || typeof w.createImageBitmap !== "function") { resolve(file); return; }
      w.createImageBitmap(file, { imageOrientation: "from-image" }).then(function (bitmap) {
        var scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
        var canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(bitmap.width * scale));
        canvas.height = Math.max(1, Math.round(bitmap.height * scale));
        canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        if (bitmap.close) { bitmap.close(); }
        canvas.toBlob(function (blob) {
          if (!blob || blob.size >= file.size) { resolve(file); return; }
          resolve(new w.File([blob], jpgName(file.name), { type: "image/jpeg" }));
        }, "image/jpeg", 0.9);
      }).catch(function () { resolve(file); });
    });
  }

  function putInInput(file) {
    try {
      var dt = new window.DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      return true;
    } catch (e) { return false; }
  }

  function handleFile(file, fromCamera) {
    var mine = ++token;
    clearPreview();
    button.disabled = true;
    if (!file) { return; }
    if (!allowedName.test(file.name) || (file.type && allowedType.indexOf(file.type) === -1)) {
      showMessage("This file type is not supported. Please choose a JPG, JPEG or PNG image.");
      return;
    }
    shrinkIfNeeded(file).then(function (f) {
      if (mine !== token) { return; }
      if (f.size > maxBytes) {
        showMessage("This file is larger than " + maxMb + " MB. Please choose a smaller image.");
        return;
      }
      if (f !== file || fromCamera) {
        if (!putInInput(f)) {
          showMessage(fromCamera ? "Your browser could not use the camera photo. Please choose a photograph instead."
                                 : "This file is larger than " + maxMb + " MB. Please choose a smaller image.");
          return;
        }
      }
      objectUrl = URL.createObjectURL(f);
      preview.src = objectUrl;
      previewBox.hidden = false;
      if (replaceButton) { replaceButton.hidden = false; }
      button.disabled = false;
    });
  }

  button.disabled = true;                       // enabled once a suitable file is chosen

  input.addEventListener("change", function () { handleFile(input.files && input.files[0], false); });
  if (replaceButton) {
    replaceButton.addEventListener("click", function () {
      token += 1;
      input.value = "";
      clearPreview();
      button.disabled = true;
      input.focus();
      input.click();
    });
  }

  /* Camera button: shown only on touch devices, where the camera opens. On a computer, choose a file instead. */
  var touch = ("ontouchstart" in window) || (window.navigator && window.navigator.maxTouchPoints > 0);
  if (camera && cameraButton && touch) {
    cameraButton.hidden = false;
    cameraButton.addEventListener("click", function () { camera.click(); });
    camera.addEventListener("change", function () {
      var file = camera.files && camera.files[0];
      handleFile(file, true);
      camera.value = "";                        // allow taking the same photograph again
    });
  }

  form.addEventListener("submit", function () {
    button.disabled = true;
    busy.hidden = false;
  });

  /* Returning with the browser's back button restores the page; clear the busy state. */
  window.addEventListener("pageshow", function (event) { if (event.persisted) { reset(); } });
})();
