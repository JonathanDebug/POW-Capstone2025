// Making a button do a simple alert
// document.getElementById("btnScan").addEventListener("click", function () {
//   alert("Hello World!");
// });


// document.getElementById('readEmail').addEventListener('click', async () => {
//   const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

//   if (tab.url.includes('outlook.office.com') || tab.url.includes('outlook.live.com')) {
//     chrome.tabs.sendMessage(tab.id, { action: "getEmailContent" }, (response) => {
//       if (response && !response.error) {
//         displayEmailData(response);
//       } else {
//         document.getElementById('emailContent').innerHTML =
//           '<p style="color: red;">Error reading email. Make sure you have an email open.</p>';
//       }
//     });
//   } else {
//     document.getElementById('emailContent').innerHTML =
//       '<p style="color: red;">Please open Outlook Web App first.</p>';
//   }
// });

document.getElementById('btnScan').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.tabs.sendMessage(tab.id, { action: "getEmailContent" }, (response) => {
    if (response && !response.error) {
      const text = `Email: ${response.email}\n`;
      console.log(text);
      navigator.clipboard.writeText(text).then(() => {
        alert('Email is supposed to be extracted');
      });
    }
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