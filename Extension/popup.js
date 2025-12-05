// Making a button do a simple alert
// document.getElementById("btnScan").addEventListener("click", function () {
//   alert("Hello World!");
// });

// When the button is clicked, send a message to content script to extract email data

document.getElementById('btnScan').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  // Force inject content.js (in case page was loaded before extension)
  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ['content.js']
  });

  // Send message to content script
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

    const text = `${response.phish ? "Possible phishing detected. Do NOT click links or download attachments" : "No phishing was detected in this email."}
${response.upr ? "This email was sent from an official UPR account." : "This message did NOT originate from an official UPR account. Be cautious with links or attachments."}
Confidence: ${response.accuracy}%
    `;

    console.log("Extracted email:", email);
    console.log("Backend analysis:", response);

    // Copy to clipboard
    navigator.clipboard.writeText(text).then(() => {
     alert(`Email scanned!\n\n${text}`);
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