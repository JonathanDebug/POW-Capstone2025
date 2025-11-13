// Making a button do a simple alert
// document.getElementById("btnScan").addEventListener("click", function () {
//   alert("Hello World!");
// });

// When the button is clicked, send a message to content script to extract email data

document.getElementById('btnScan').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.tabs.sendMessage(tab.id, { action: "getEmailContent" }, (response) => {
    if (!response) {
      alert("No response — content script might not be injected.");
      return;
    }

    if (!response.success) {
      alert("Error: " + response.error);
      return;
    }

    const email = response.email;
    const analysis = response.analysis;

    console.log("Extracted email:", email);
    console.log("Backend analysis:", analysis);

    const text = `
Subject: ${email.subject}
Sender:  ${email.sender}
Body:    ${email.body.substring(0, 200)}...

Prediction: ${analysis.label}
Score: ${analysis.score}
    `;

    navigator.clipboard.writeText(text).then(() => {
      alert(`Email scanned.\nPrediction: ${analysis.label}`);
    });
  });
});


// function displayEmailData(emailData) {
//   const contentDiv = document.getElementById('emailContent');
//   contentDiv.innerHTML = `
//     <div class="email-data">
//       <div class="subject">${escapeHtml(emailData.subject)}</div>
//       <div class="sender">From: ${escapeHtml(emailData.sender)}</div>
//       <div class="body">${escapeHtml(emailData.body.substring(0, 500))}${emailData.body.length > 500 ? '...' : ''}</div>
//     </div>
//   `;
// }

// function escapeHtml(unsafe) {
//   return unsafe
//     .replace(/&/g, "&amp;")
//     .replace(/</g, "&lt;")
//     .replace(/>/g, "&gt;")
//     .replace(/"/g, "&quot;")
//     .replace(/'/g, "&#039;");
// }