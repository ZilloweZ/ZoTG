// Import required modules
const transformers = require('@huggingface/transformers');
const { GPT2LMHeadModel, GPT2Tokenizer } = transformers;
const { GPTNeoForCausalLM, GPT2TokenizerFast } = transformers;

// Load the models and tokenizers
const tokenizerGPT2 = new GPT2Tokenizer.fromPretrained('gpt2');
const modelGPT2 = new GPT2LMHeadModel.fromPretrained('gpt2');

const tokenizerGPTNeo = new GPT2TokenizerFast.fromPretrained('EleutherAI/gpt-neo-1.3B');
const modelGPTNeo = new GPTNeoForCausalLM.fromPretrained('EleutherAI/gpt-neo-1.3B');

// Get the HTML elements
const messageInput = document.getElementById('message');
const chatLog = document.getElementById('chatlog');
const modelSelect = document.getElementById('model');

// Function to generate a response based on the selected model
async function generateResponse(prompt) {
  let tokenizer, model;
  if (modelSelect.value === 'gpt2') {
    tokenizer = tokenizerGPT2;
    model = modelGPT2;
  } else if (modelSelect.value === 'gptneo') {
    tokenizer = tokenizerGPTNeo;
    model = modelGPTNeo;
  }

  const inputIds = tokenizer.encode(prompt);
  const output = await model.generate(inputIds, { max_length: 1000 });
  const response = tokenizer.decode(output[0], { skipSpecialTokens: true });
  return response;
}

// Function to add a message to the chat log
function addMessageToLog(message, isUser = false) {
  const messageElement = document.createElement('div');
  messageElement.classList.add('message');
  if (isUser) {
    messageElement.classList.add('user-message');
  } else {
    messageElement.classList.add('bot-message');
  }
  messageElement.innerText = message;
  chatLog.appendChild(messageElement);
}

// Function to send a message and receive a response
async function sendMessage() {
  const message = messageInput.value;
  if (message) {
    addMessageToLog(message, true);
    messageInput.value = '';
    const response = await generateResponse(message);
    addMessageToLog(response);
  }
}

// Add an event listener to the Send button
const sendButton = document.querySelector('.chat-form button');
sendButton.addEventListener('click', sendMessage);

// Add an event listener to the message input to allow submitting with Enter key
messageInput.addEventListener('keyup', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    sendButton.click();
  }
});
