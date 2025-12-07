// Making a button do a simple alert
// document.getElementById("btnScan").addEventListener("click", function () {
//   alert("Hello World!");
// });

// When the button is clicked, send a message to content script to extract email data

document.getElementById('btnScan').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  // Check if current tab URL contains "outlook"
  if (!tab.url.includes("outlook")) {
    const scanSection = document.querySelector(".scan-section");
    scanSection.innerHTML = `
      <div style="font-size: 13px; color: #d93025; margin-bottom: 14px; line-height: 1.4; text-align:center;">
        This extension only works on Outlook.
      </div>
      <button id="goToOutlook" class="scan-button">Go to Outlook</button>
    `;

    // Add click event to open Outlook in a new tab
    document.getElementById("goToOutlook").addEventListener("click", () => {
      chrome.tabs.create({ url: "https://www.microsoft.com/en-us/microsoft-365/outlook/log-in" });
    });

    return; // Stop further execution
  }

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

    // Build styled results
    const scanSection = document.querySelector(".scan-section");
    scanSection.innerHTML = `
      <div style="font-size: 13px; line-height: 1.4; margin-bottom: 14px; text-align:center;">
        <div style="font-weight:600; margin-bottom:6px;">
          ${response.phish ?
            '<span style="color:#d93025;">Possible phishing detected</span>' :
            '<span style="color:#188038;"> No phishing detected</span>'}
        </div>
        <div>Confidence: ${response.accuracy}%</div>
        <div style="font-weight:600; margin-bottom:6px;">
          ${response.phish ?
            '<span style="color:#d93025;">The AI model detected patterns similar to those found in phishing messages</span>' :
            '<span style="color:#188038;"> The AI model did NOT detect patterns found in phishing in this message</span>'}
        </div>
        <div style="margin-bottom:6px;">
          ${response.upr ?
            '<span style="color:#188038;">This email was sent from an official UPR account</span>' :
            '<span style="color:#d93025;">This email was NOT sent from an official UPR account</span>'}
        </div>

      </div>
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