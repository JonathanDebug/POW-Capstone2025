console.log("Content script loaded!");

// Define this first
function extractEmailData() {
  try {
    const subject = document.querySelector(".f77rj")?.textContent || 'Sender not found';
    const sender = document.querySelector('span.OZZZK')?.textContent || 'Sender not found';
    const body = document.querySelector('[aria-label="Message body"]')?.innerText || 'Body not found';

    return {
      subject: subject.trim(),
      sender: sender.trim(),
      body: body.replace(/\n{2,}/g, '\n').replace(/\s{2,}/g, ' ').trim(),
      timestamp: new Date().toISOString(),
      url: window.location.href
    };
  } catch (error) {
    console.error('Error extracting email data:', error);
    return { error: error.message };
  }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getEmailContent") {
    const emailData = extractEmailData();

    fetch("http://localhost:8000/analyze-email", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(emailData)
    })
      .then(res => res.json())
      .then(analysis => {
        sendResponse({
          success: true,
          email: emailData,
          phish: analysis.phish,
          accuracy: analysis.accuracy,
          upr: analysis.upr
        });
      })
      .catch(err => {
        sendResponse({
          success: false,
          error: err.message
        });
      });

    return true; // keep async channel open
  }
});

