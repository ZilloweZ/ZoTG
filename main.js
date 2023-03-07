// main.js

// Import the required modules from the transformers library
const { pipeline } = require('@huggingface/sentence-transformers');

// Define the chatbot function
async function chatbot(message) {
  // Define the input and output formats for the transformer model
  const config = {
    inputFormat: 'plain',
    outputFormat: 'plain',
  };

  // Load the transformer model
  const model = await pipeline({
    model: 'microsoft/DialoGPT-large',
    ...config,
  });

  // Generate a response to the input message using the transformer model
  const response = await model(message);

  return response;
}

// Get the chat form and chatlogs elements
const chatForm = document.querySelector('.chat-form');
const chatlogs = document.querySelector('.chatlogs');

// Add an event listener to the chat form submit button
chatForm.addEventListener('submit', async (event) => {
  // Prevent the form from being submitted
  event.preventDefault();

  // Get the input message from the chat form input element
  const message = event.target.elements.message.value;

  // If the input message is not empty
  if (message.trim() !== '') {
    // Add the user's message to the chatlogs
    chatlogs.innerHTML += `
      <div class="chat">
        <div class="user-icon">👤</div>
        <div class="user-message">${message}</div>
      </div>
    `;

    // Generate a response to the input message using the chatbot function
    const response = await chatbot(message);

    // Add the chatbot's response to the chatlogs
    chatlogs.innerHTML += `
      <div class="chat">
        <div class="bot-icon">🤖</div>
        <div class="bot-message">${response}</div>
      </div>
    `;

    // Clear the chat form input element
    event.target.elements.message.value = '';
  }
});
