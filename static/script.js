const stepUpload = document.getElementById('step-upload');
const stepPreview = document.getElementById('step-preview');
const groupsContainer = document.getElementById('groups-container');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const uploadStatus = document.getElementById('upload-status');
const maskStatus = document.getElementById('mask-status');
const instructionsInput = document.getElementById('instructions');
const maskBtn = document.getElementById('mask-btn');
const backBtn = document.getElementById('back-btn');

const previewSubhead = document.getElementById('preview-subhead');
const previewImage = document.getElementById('preview-image');
const previewBoxesLayer = document.getElementById('preview-boxes');
const previewCanvas = document.getElementById('preview-canvas');
const previewPrevBtn = document.getElementById('preview-prev');
const previewNextBtn = document.getElementById('preview-next');
const previewPageLabel = document.getElementById('preview-page-label');
const undoBtn = document.getElementById('undo-btn');
const redoBtn = document.getElementById('redo-btn');

let currentJobId = null;
let selectedGroupIds = new Set();     // whole-group checkbox selections (field list)
let selectedInstanceIds = new Set();  // individual instance selections (preview clicks)
let manualBoxes = [];                 // [{ page, bbox: [l, t, r, b] }] hand-drawn in preview
let manualUndoStack = [];             // snapshots of manualBoxes taken before each edit
let manualRedoStack = [];
let allInstances = [];                // every detected instance, from /extract/status
let groupsById = {};                  // group_id -> group data (for instance_ids lookup)
let numPages = 1;
let pageSizes = [];                   // [{w, h}] per page, in source pixel coords
let currentPage = 0;

function setStatus(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle('status--error', isError);
}

dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    fileName.textContent = file.name;
    uploadFile(file);
  }
});

function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  setStatus(uploadStatus, 'Uploading and processing PDF…');
  fetch('/extract', { method: 'POST', body: formData })
    .then((res) => res.json())
    .then((data) => {
      if (data.job_id) {
        currentJobId = data.job_id;
        pollJobStatus(data.job_id);
      } else {
        setStatus(uploadStatus, data.error || 'Upload failed', true);
      }
    })
    .catch((err) => setStatus(uploadStatus, err.message || 'Upload failed', true));
}

function pollJobStatus(jobId) {
  fetch(`/extract/status/${jobId}`)
    .then((res) => res.json())
    .then((data) => {
      if (data.status === 'processing') {
        setStatus(uploadStatus, 'Scanning document and extracting fields…');
        setTimeout(() => pollJobStatus(jobId), 1000);
        return;
      }
      if (data.status === 'error') {
        setStatus(uploadStatus, data.error || 'Processing failed', true);
        return;
      }
      if (data.status === 'done') {
        const groups = Array.isArray(data.groups) ? data.groups : (data.extra && Array.isArray(data.extra.groups) ? data.extra.groups : []);
        const instances = Array.isArray(data.instances) ? data.instances : (data.extra && Array.isArray(data.extra.instances) ? data.extra.instances : []);
        const sizes = Array.isArray(data.page_sizes) ? data.page_sizes : (data.extra && Array.isArray(data.extra.page_sizes) ? data.extra.page_sizes : []);
        const message = data.message || (data.extra && data.extra.message) || '';
        numPages = data.num_pages || (data.extra && data.extra.num_pages) || 1;
        pageSizes = sizes;
        allInstances = instances;
        renderGroups(groups);
        const fieldsMsg = message || 'Select what to mask below, or draw a box on the preview.';
        previewSubhead.textContent = `${numPages} page(s) scanned. ${fieldsMsg}`;
        initPreview();
        stepUpload.hidden = true;
        stepPreview.hidden = false;
        setStatus(uploadStatus, '');
      }
    })
    .catch((err) => setStatus(uploadStatus, err.message || 'Processing failed', true));
}

function renderGroups(groups) {
  const groupList = Array.isArray(groups) ? groups : [];
  groupsContainer.innerHTML = '';
  selectedGroupIds.clear();
  selectedInstanceIds.clear();
  groupsById = {};
  groupList.forEach((g) => { groupsById[g.group_id] = g; });
  if (!groupList.length) {
    groupsContainer.innerHTML = '<p class="empty-state">No fields detected automatically. You can still use the description box below, or draw boxes directly on the preview.</p>';
    return;
  }
  const frag = document.createDocumentFragment();
  groupList.forEach((group) => {
    const card = document.createElement('label');
    card.className = 'group-card';
    card.innerHTML = `
      <input type="checkbox" class="group-checkbox" value="${group.group_id}">
      <div class="group-card__content">
        <div class="group-card__title">${group.display_label}</div>
        <div class="group-card__meta">${group.category_label} · ${group.count} found</div>
        <div class="group-card__samples">${(group.sample_values || []).slice(0, 3).join(' · ')}</div>
      </div>
    `;
    const input = card.querySelector('input');
    input.addEventListener('change', () => {
      setGroupSelected(group, input.checked);
      renderPreviewBoxes();
    });
    frag.appendChild(card);
  });
  groupsContainer.appendChild(frag);
}

function setGroupSelected(group, selected) {
  if (selected) {
    selectedGroupIds.add(group.group_id);
    (group.instance_ids || []).forEach((id) => selectedInstanceIds.add(id));
  } else {
    selectedGroupIds.delete(group.group_id);
    (group.instance_ids || []).forEach((id) => selectedInstanceIds.delete(id));
  }
}

function groupCheckbox(groupId) {
  return groupsContainer.querySelector(`.group-checkbox[value="${CSS.escape(groupId)}"]`);
}

function groupIdForInstance(inst) {
  return `${inst.category}::${inst.field_type}::${inst.display_label}`;
}

// Keeps the field-list checkbox for an instance's group in sync
// (checked/unchecked/indeterminate) after a box is toggled individually
// in the preview, so the two selection UIs never disagree.
function syncGroupCheckboxFor(inst) {
  const groupId = groupIdForInstance(inst);
  const group = groupsById[groupId];
  const checkbox = groupCheckbox(groupId);
  if (!group || !checkbox) return;
  const ids = group.instance_ids || [];
  const selectedCount = ids.filter((id) => selectedInstanceIds.has(id)).length;
  checkbox.checked = selectedCount === ids.length && ids.length > 0;
  checkbox.indeterminate = selectedCount > 0 && selectedCount < ids.length;
  if (checkbox.checked) {
    selectedGroupIds.add(groupId);
  } else {
    selectedGroupIds.delete(groupId);
  }
}

// ---- Preview tab ----

function initPreview() {
  currentPage = 0;
  manualBoxes = [];
  manualUndoStack = [];
  manualRedoStack = [];
  updateUndoRedoButtons();
  renderPreviewPage();
}

function renderPreviewPage() {
  previewPageLabel.textContent = `Page ${currentPage + 1} of ${numPages}`;
  previewPrevBtn.disabled = currentPage === 0;
  previewNextBtn.disabled = currentPage >= numPages - 1;
  previewImage.onload = renderPreviewBoxes;
  previewImage.src = `/jobs/${currentJobId}/page/${currentPage}`;
}

previewPrevBtn.addEventListener('click', () => {
  if (currentPage > 0) { currentPage -= 1; renderPreviewPage(); }
});
previewNextBtn.addEventListener('click', () => {
  if (currentPage < numPages - 1) { currentPage += 1; renderPreviewPage(); }
});
// ---- Undo/redo for hand-drawn ("custom redaction") boxes only ----
// Detected-field selections aren't part of this history — only the
// boxes the user drew themselves, matching the sidebar's own scope.

function snapshotManualBoxes() {
  return manualBoxes.map((b) => ({ page: b.page, bbox: b.bbox.slice() }));
}

function pushManualUndoSnapshot() {
  manualUndoStack.push(snapshotManualBoxes());
  manualRedoStack = [];
  updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
  undoBtn.disabled = manualUndoStack.length === 0;
  redoBtn.disabled = manualRedoStack.length === 0;
}

undoBtn.addEventListener('click', () => {
  if (!manualUndoStack.length) return;
  manualRedoStack.push(snapshotManualBoxes());
  manualBoxes = manualUndoStack.pop();
  updateUndoRedoButtons();
  renderPreviewBoxes();
});

redoBtn.addEventListener('click', () => {
  if (!manualRedoStack.length) return;
  manualUndoStack.push(snapshotManualBoxes());
  manualBoxes = manualRedoStack.pop();
  updateUndoRedoButtons();
  renderPreviewBoxes();
});

function pageSize() {
  return pageSizes[currentPage] || { w: previewImage.naturalWidth || 1, h: previewImage.naturalHeight || 1 };
}

function renderPreviewBoxes() {
  previewBoxesLayer.innerHTML = '';
  const { w, h } = pageSize();
  if (!w || !h) return;

  allInstances
    .filter((inst) => inst.page === currentPage)
    .forEach((inst) => {
      const [left, top, right, bottom] = inst.bbox;
      const el = document.createElement('div');
      el.className = 'preview-box' + (selectedInstanceIds.has(inst.id) ? ' preview-box--selected' : '');
      el.style.left = `${(left / w) * 100}%`;
      el.style.top = `${(top / h) * 100}%`;
      el.style.width = `${((right - left) / w) * 100}%`;
      el.style.height = `${((bottom - top) / h) * 100}%`;
      const label = document.createElement('span');
      label.className = 'preview-box__label';
      label.textContent = inst.display_label;
      el.appendChild(label);
      el.addEventListener('click', (evt) => {
        evt.stopPropagation();
        if (selectedInstanceIds.has(inst.id)) {
          selectedInstanceIds.delete(inst.id);
        } else {
          selectedInstanceIds.add(inst.id);
        }
        syncGroupCheckboxFor(inst);
        renderPreviewBoxes();
      });
      previewBoxesLayer.appendChild(el);
    });

  manualBoxes
    .map((box, idx) => ({ box, idx }))
    .filter(({ box }) => box.page === currentPage)
    .forEach(({ box, idx }) => {
      const [left, top, right, bottom] = box.bbox;
      const el = document.createElement('div');
      el.className = 'preview-box preview-box--manual';
      el.style.left = `${(left / w) * 100}%`;
      el.style.top = `${(top / h) * 100}%`;
      el.style.width = `${((right - left) / w) * 100}%`;
      el.style.height = `${((bottom - top) / h) * 100}%`;
      const del = document.createElement('span');
      del.className = 'preview-box__delete';
      del.textContent = '×';
      del.addEventListener('click', (evt) => {
        evt.stopPropagation();
        pushManualUndoSnapshot();
        manualBoxes.splice(idx, 1);
        renderPreviewBoxes();
      });
      el.appendChild(del);
      previewBoxesLayer.appendChild(el);
    });
}

// ---- Drawing new boxes on the preview canvas ----

let dragState = null;

previewCanvas.addEventListener('mousedown', (evt) => {
  if (evt.target !== previewImage && evt.target !== previewCanvas) return; // ignore clicks starting on an existing box
  const rect = previewCanvas.getBoundingClientRect();
  dragState = {
    startX: evt.clientX - rect.left,
    startY: evt.clientY - rect.top,
    rectEl: document.createElement('div'),
  };
  dragState.rectEl.className = 'preview-drag-rect';
  previewBoxesLayer.appendChild(dragState.rectEl);
});

previewCanvas.addEventListener('mousemove', (evt) => {
  if (!dragState) return;
  const rect = previewCanvas.getBoundingClientRect();
  const x = evt.clientX - rect.left;
  const y = evt.clientY - rect.top;
  const left = Math.min(x, dragState.startX);
  const top = Math.min(y, dragState.startY);
  const width = Math.abs(x - dragState.startX);
  const height = Math.abs(y - dragState.startY);
  Object.assign(dragState.rectEl.style, {
    left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px`,
  });
});

window.addEventListener('mouseup', (evt) => {
  if (!dragState) return;
  const rect = previewCanvas.getBoundingClientRect();
  const x = Math.min(Math.max(evt.clientX - rect.left, 0), rect.width);
  const y = Math.min(Math.max(evt.clientY - rect.top, 0), rect.height);
  const left = Math.min(x, dragState.startX);
  const top = Math.min(y, dragState.startY);
  const right = Math.max(x, dragState.startX);
  const bottom = Math.max(y, dragState.startY);
  dragState.rectEl.remove();
  dragState = null;

  if (right - left < 6 || bottom - top < 6 || rect.width === 0 || rect.height === 0) return; // treat as a click, not a drag

  pushManualUndoSnapshot();
  const { w, h } = pageSize();
  manualBoxes.push({
    page: currentPage,
    bbox: [
      (left / rect.width) * w,
      (top / rect.height) * h,
      (right / rect.width) * w,
      (bottom / rect.height) * h,
    ],
  });
  renderPreviewBoxes();
});

maskBtn.addEventListener('click', () => {
  if (!currentJobId) {
    setStatus(maskStatus, 'Please upload a document first.', true);
    return;
  }
  if (selectedInstanceIds.size === 0 && manualBoxes.length === 0 && !instructionsInput.value.trim()) {
    setStatus(maskStatus, 'Select a field, draw a box, or enter a description to mask.', true);
    return;
  }

  setStatus(maskStatus, 'Masking document…');
  fetch('/mask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      job_id: currentJobId,
      instance_ids: Array.from(selectedInstanceIds),
      manual_boxes: manualBoxes,
      instructions: instructionsInput.value.trim(),
    }),
  })
    .then((res) => {
      if (!res.ok) {
        return res.json().then((data) => Promise.reject(new Error(data.error || 'Masking failed')));
      }
      return res.blob().then((blob) => ({ blob }));
    })
    .then(({ blob }) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'masked_output.pdf';
      a.click();
      URL.revokeObjectURL(url);
      setStatus(maskStatus, 'Document masked successfully.');
    })
    .catch((err) => setStatus(maskStatus, err.message || 'Masking failed', true));
});

backBtn.addEventListener('click', () => {
  stepPreview.hidden = true;
  stepUpload.hidden = false;
  groupsContainer.innerHTML = '';
  selectedGroupIds.clear();
  selectedInstanceIds.clear();
  manualBoxes = [];
  manualUndoStack = [];
  manualRedoStack = [];
  updateUndoRedoButtons();
  allInstances = [];
  groupsById = {};
  instructionsInput.value = '';
  setStatus(uploadStatus, '');
  setStatus(maskStatus, '');
  currentJobId = null;
  fileInput.value = '';
  fileName.textContent = '';
});
