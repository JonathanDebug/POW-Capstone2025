// Making a button do a simple alert
// document.getElementById("btnScan").addEventListener("click", function () {
//   alert("Hello World!");
// });

// When the button is clicked, send a message to content script to extract email data

document.getElementById('btnScan').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.tabs.sendMessage(tab.id, { action: "getEmailContent" }, (response) => {
    if (response && !response.error) {
      
      const emailData = {
        subject: response.subject,
        from: response.sender,
        body: response.body,
      };

      // Create and download the JSON file
      const dataStr = JSON.stringify(emailData, null, 2);
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      
      const emaildata = document.createElement('a');
      emaildata.href = url;
      emaildata.download = 'POW Database.json';
      document.body.appendChild(emaildata);
      emaildata.click();
      document.body.removeChild(emaildata);
      URL.revokeObjectURL(url);
      
      alert('POW Database.json has been downloaded!!');
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