// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getEmailContent") {
    const emailData = extractEmailData();
    sendResponse(emailData);
  }
  return true;
});

function extractEmailData() {
  try {
    // For Outlook Web App
    // querySelector will look for the firs element that has [ária-label="Message subject"].
    // This however is not how it is in the OWA.
    const Email = document.querySelector('[aria-label="Email message"]')?.textContent ||
      'Subject not found';
    // const sender = document.querySelector('[aria-label="Message sender"]')?.textContent ||
    //   document.querySelector('[email-type="from"]')?.textContent ||
    //   'Sender not found';

    // const body = document.querySelector('[role="document"]')?.textContent ||
    //   document.querySelector('.allowTextSelection.bodyContents')?.textContent ||
    //   document.querySelector('.x_body')?.textContent ||
    //   'Body not found';

    return {
      email: Email.trim(),
      // subject: subject.trim(),
      // sender: sender.trim(),
      // body: body.trim(),
      timestamp: new Date().toISOString(),
      url: window.location.href
    };
  } catch (error) {
    console.error('Error extracting email data:', error);
    return { error: error.message };
  }
}

// // Alternative approach: Monitor DOM changes for single-page app
// const observer = new MutationObserver(() => {
//   // Re-extract data when DOM changes (email navigation)
//   const emailData = extractEmailData();
//   if (emailData.body && emailData.body !== 'Body not found') {
//     chrome.storage.local.set({ currentEmail: emailData });
//   }
// });

// observer.observe(document.body, {
//   childList: true,
//   subtree: true
// });