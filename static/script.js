const stepUpload = document.getElementById('step-upload');
const stepReview = document.getElementById('step-review');
const groupsContainer = document.getElementById('groups-container');
const reviewSubhead = document.getElementById('review-subhead');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const uploadStatus = document.getElementById('upload-status');
const maskStatus = document.getElementById('mask-status');
const instructionsInput = document.getElementById('instructions');
const maskBtn = document.getElementById('mask-btn');
const backBtn = document.getElementById('back-btn');

let currentJobId = null;
let selectedGroupIds = new Set();

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
        const message = data.message || (data.extra && data.extra.message) || '';
        renderGroups(groups, message);
        stepUpload.hidden = true;
        stepReview.hidden = false;
        setStatus(uploadStatus, '');
      }
    })
    .catch((err) => setStatus(uploadStatus, err.message || 'Processing failed', true));
}

function renderGroups(groups, message) {
  const groupList = Array.isArray(groups) ? groups : [];
  groupsContainer.innerHTML = '';
  selectedGroupIds.clear();
  reviewSubhead.textContent = message || 'Select what to mask. Fields are grouped by type — checking one masks every occurrence.';
  if (!groupList.length) {
    groupsContainer.innerHTML = '<p class="empty-state">No fields detected automatically. You can still use the description box below.</p>';
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
      if (input.checked) {
        selectedGroupIds.add(group.group_id);
      } else {
        selectedGroupIds.delete(group.group_id);
      }
    });
    frag.appendChild(card);
  });
  groupsContainer.appendChild(frag);
}

maskBtn.addEventListener('click', () => {
  if (!currentJobId) {
    setStatus(maskStatus, 'Please upload a document first.', true);
    return;
  }
  if (selectedGroupIds.size === 0 && !instructionsInput.value.trim()) {
    setStatus(maskStatus, 'Select a field or enter a description to mask.', true);
    return;
  }

  setStatus(maskStatus, 'Masking document…');
  fetch('/mask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      job_id: currentJobId,
      group_ids: Array.from(selectedGroupIds),
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
  stepReview.hidden = true;
  stepUpload.hidden = false;
  groupsContainer.innerHTML = '';
  selectedGroupIds.clear();
  instructionsInput.value = '';
  reviewSubhead.textContent = 'Select what to mask. Fields are grouped by type — checking one masks every occurrence.';
  setStatus(uploadStatus, '');
  setStatus(maskStatus, '');
  currentJobId = null;
  fileInput.value = '';
  fileName.textContent = '';
});
