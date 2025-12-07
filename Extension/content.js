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


    //subject in the dom is named f77rj
    const subject = document.querySelector(".f77rj")?.textContent ||
      'Sender not found';
    //sender in the dom is named OZZZK, but it's a span
    const sender = document.querySelector('span.OZZZK')?.textContent ||
      'Sender not found';

    //body in the dom we can look it up with the aria-label "Message body"
    const body = document.querySelector('[role="document"]')?.innerText ||
      'Body not found';


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

