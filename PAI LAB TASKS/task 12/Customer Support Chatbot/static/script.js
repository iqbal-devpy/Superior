document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const clearChatBtn = document.getElementById("clear-chat");
    const themeToggle = document.getElementById("theme-toggle");
    const typingIndicator = document.getElementById("typing-indicator");
    const notificationSound = new Audio("static/ping.mp3");

    function appendMessage(sender, text) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("chat-message", sender === "user" ? "user-message" : "bot-message");
        messageDiv.textContent = text;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        
        if (sender === "bot") notificationSound.play();
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        appendMessage("user", message);
        userInput.value = "";

        typingIndicator.style.display = "block";

        fetch("http://127.0.0.1:5000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: message }),
        })
        .then(response => response.json())
        .then(data => {
            typingIndicator.style.display = "none";
            appendMessage("bot", data.response);
        })
        .catch(() => {
            typingIndicator.style.display = "none";
            appendMessage("bot", "Error: Could not connect to chatbot.");
        });
    }

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") sendMessage();
    });

    clearChatBtn.addEventListener("click", () => { chatBox.innerHTML = ""; });
    themeToggle.addEventListener("click", () => {
        document.body.classList.toggle("light-mode");
        themeToggle.textContent = document.body.classList.contains("light-mode") ? "🌙" : "☀️";
    });
});
